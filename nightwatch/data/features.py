"""Define feature extractors."""
import numpy as np
import pandas as pd

from scipy.stats import norm

from nightwatch.data.common import SleepStage, UserData, WearableFeatures


def get_valid_psg(user_data: UserData, start_time: int, frame_size:
                  int = 30):
    """Return valid PSG label.

    A PSG label is valid if it has corresponding motion and heartrate
    samples.

    Args:
    user_data (UserData): raw user samples.
    start_time (int): starting timestamp in seconds.
    frame_size (int): sampling interval in seconds.

    Returns:
    a pd.DataFrame containing valid timestamps and PSG labels.

    """
    # find overlapping timestamps
    rel_motion_ts = (
        user_data.motion.timestamp - \
        (user_data.motion.timestamp - start_time) % frame_size
    ).unique()
    rel_hr_ts = (
        user_data.heartrate.timestamp -
        (user_data.heartrate.timestamp - start_time) % frame_size
    ).unique()

    timestamps_with_samples = np.intersect1d(rel_motion_ts, rel_hr_ts)
    psg_samples = user_data.psg[
        user_data.psg.timestamp.isin(timestamps_with_samples)
    ]
    
    # interpolate unscored samples
    psg_samples.loc[:, "label"] = psg_samples.label.replace(
            SleepStage.unscored.value, np.nan
        )
    psg_samples.loc[:, "label"] = psg_samples.label.interpolate().astype(int)
    return psg_samples


def convolve_with_dog(y, box_pts):
    # from https://github.com/ojwalch/sleep_classifiers/
    y = y - np.mean(y)
    box = np.ones(box_pts) / box_pts

    mu1 = int(box_pts / 2.0)
    sigma1 = 120

    mu2 = int(box_pts / 2.0)
    sigma2 = 600

    scalar = 0.75

    for ind in range(0, box_pts):
        box[ind] = np.exp(-1 / 2 * (((ind - mu1) / sigma1) ** 2)) - scalar * np.exp(
            -1 / 2 * (((ind - mu2) / sigma2) ** 2))

    y = np.insert(y, 0, np.flip(y[0:int(box_pts / 2)]))  # Pad by repeating boundary conditions
    y = np.insert(y, len(y) - 1, np.flip(y[int(-box_pts / 2):]))
    y_smooth = np.convolve(y, box, mode='valid')

    return y_smooth


class WearableFeatureExtractor:
    """Compute features for heart rate and motion data."""
    
    def __init__(self, window_size: int = 50):
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = window_size

    def __call__(self, user_data: UserData) -> WearableFeatures:
        """Extract features from user data."""
        if user_data is None:
            raise ValueError("user_data is None")

        start_time = user_data.psg.timestamp.min()
        valid_psg = get_valid_psg(user_data, start_time)

        cos_time, real_time = self._generate_time(valid_psg.timestamp)
        
        return WearableFeatures(
            motion=self._compute_activity_features(user_data, valid_psg.timestamp),
            heartrate=self._compute_hr_features(user_data, valid_psg.timestamp),
            cos_time=cos_time,
            real_time=real_time,
            labels=np.array(valid_psg.label)
        )
            
    def _compute_activity_features(self, user_data, psg_timestamps):
        df = user_data.activity_count.copy()
        df["timestamp"] = df.timestamp.astype(np.int32)

        # make sure the sampling rate is consistent. 
        # add missing timestamps and interpolate activity count.
        sample_ts = pd.DataFrame({'timestamp': np.arange(df.index.min(), df.index.max() + 1, 1)})
        resampled = pd.merge(sample_ts, df, on='timestamp', how='left')
        resampled['activity_count'] = resampled['activity_count'].interpolate()
        df = resampled

        # the activity signal is convolved with a normalized gaussian
        # filter with mu=window_size/2 and sigma=50 (s)
        mu = self.window_size / 2
        sigma = 50
        kern = norm.pdf(np.arange(self.window_size), mu, sigma)
        kern = kern / kern.sum()
        results = []

        # Perform the operations
        for curr_s in psg_timestamps[1:]:
            # find the closest index to curr_s
            idx = (np.abs(df['timestamp'] - curr_s)).argmin()

            # select self.window_size past samples ending at idx
            window_data = df['activity_count'][
                 max(0, idx - self.window_size + 1): idx
            ]
            actual_size = window_data.shape[0]
            if actual_size != self.window_size:
                # pad window with zeros
                padded_window = np.zeros(self.window_size)
                padded_window[self.window_size - actual_size:] = window_data
                window_data = padded_window

            # convolve with the gaussian kern and add
            results.append(np.dot(window_data, kern))

        return np.array(results)

    def _compute_hr_features(self, user_data, psg_timestamps):
        df = user_data.heartrate.copy()
        df["timestamp"] = df.timestamp.astype(np.int32)

        # make sure the sampling rate is consistent. 
        # add missing timestamps and interpolate heart rate.
        hr_mean = df.heartrate.mean()
        sample_ts = pd.DataFrame({'timestamp': np.arange(df.timestamp.min(), df.timestamp.max() + 1, 1)})
        resampled = pd.merge(sample_ts, df, on='timestamp', how='left')
        resampled['heartrate'] = resampled['heartrate'].interpolate()
        df = resampled

        # smooth and normalize
        num_samples = len(df.heartrate)
        df["heartrate"] = df.heartrate.fillna(hr_mean)
        df["heartrate"] = convolve_with_dog(df.heartrate, 300)[:num_samples]
        scalar = np.percentile(np.abs(df.heartrate), 90)
        df["heartrate"] = df.heartrate / scalar

        results = []

        # Perform the operations
        for curr_s in psg_timestamps[1:]:
            # find the closest index to curr_s
            idx = (np.abs(df['timestamp'] - curr_s)).argmin()

            # select window_size past samples ending at curr_s
            window_data = df['heartrate'][
                 max(0, idx - self.window_size + 1): idx
            ]
            actual_size = window_data.shape[0]
            if actual_size != self.window_size:
                # pad window with zeros
                padded_window = np.zeros(self.window_size)
                padded_window[self.window_size - actual_size:] = window_data
                window_data = padded_window

            results.append(np.std(window_data))

        return np.array(results)

    def _generate_time(self, psg_timestamps, shift_h: int = 5):
        clock = lambda t: -np.math.cos(
            2 * np.math.pi / (24 * 3600) *
            (t - shift_h * 3600) * 3600
        )
        
        cos_time = []
        time = []
        t0 = psg_timestamps[0]

        for curr_t in psg_timestamps:
            cos_time.append(clock(curr_t - t0))
            time.append((curr_t - t0) / 3600.)

        return np.array(cos_time), np.array(time)

