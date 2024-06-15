"""Sleep-Accel dataset processing functions."""
from typing import List
from datetime import datetime
from pathlib import Path
from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt



def _compute_activity_counts(acc_df: pd.DataFrame, fs: int = 50) -> pd.DataFrame:
    """Compute activity counts from user acc samples."""
    
    # Adapted from:
    # https://github.com/ojwalch/sleep_classifiers/tree/main/source/preprocessing/activity_count
    start_ts, end_ts = acc_df.timestamp.min(), acc_df.timestamp.max()
    time = np.arange(np.amin(start_ts, end_ts, 1.0 / fs))
    z_data = np.interp(time, acc_df.timestamp, acc_df.acc_z)

    b, a = butter(order, [3 / (fs / 2), 11 / (fs / 2)], 'bandpass')
    z_filt = filtfilt(b, a, z_data)
    z_filt = np.abs(z_filt)
    
    top_edge = 5
    bottom_edge = 0
    number_of_bins = 128

    bin_edges = np.linspace(bottom_edge, top_edge, number_of_bins + 1)
    binned = np.digitize(z_filt, bin_edges)
    
    epoch = 15
    counts = ActivityCountService.max2epochs(binned, fs, epoch)
    counts = (counts - 18) * 3.07
    counts[counts < 0] = 0

    time_counts = np.linspace(np.min(data[:, 0]), max(data[:, 0]), np.shape(counts)[0])
    time_counts = np.expand_dims(time_counts, axis=1)
    counts = np.expand_dims(counts, axis=1)
    output = np.hstack((time_counts, counts))


UserData = namedtuple("UserData", ["labels", "motion", "heartrate", "activity_count"])

class SleepAccel:

    def __init__(self, ds_dir: str):
        """Initialize for a given dataset dir."""
        
        if ds_dir is None:
            raise ValueError("ds_path is None")
        
        self._ds_path = Path(ds_dir)
        if not self._ds_path.exists() or not self._ds_path.is_dir():
            raise IOError(f"could not find dir {ds_dir}")

        self.user_ids = [p.stem for p in self._ds_path.glob("*") if p.is_dir()]
        

    def _load_user_file(self, user_id: str, fname: str,
                        cols: List[str], delim = ",") -> pd.DataFrame:
        if user_id is None or user_id not in  self.user_ids:
            raise ValueError("must specify a valid user id")
        if fname is None:
            raise ValueError("the filename is not specified.")
        
        fpath = self._ds_path / user_id / fname
        if not fpath.exists() or not fpath.is_file():
            raise IOError(f"could not load {fpath}")
        
        return pd.read_csv(fpath.as_posix(), 
                           header=None, 
                           delimiter=delim, 
                           names=cols)

    def _compute_activity_counts(self, acc_df: pd.DataFrame,
                                 fs: int = 50, frame_size: int = 15) -> pd.DataFrame:
        """Compute activity counts from user acc samples."""

        # Adapted from:
        # https://github.com/ojwalch/sleep_classifiers/tree/main/source/preprocessing/activity_count
        start_ts, end_ts = acc_df.timestamp.min(), acc_df.timestamp.max()
        time = np.arange(start_ts, end_ts, 1.0 / fs)
        z_data = np.interp(time, acc_df.timestamp, acc_df.acc_z)

        b, a = butter(5, [3 / (fs / 2), 11 / (fs / 2)], 'bandpass')
        z_filt = filtfilt(b, a, z_data)
        z_filt = np.abs(z_filt)

        num_bins = 128
        bin_edges = np.linspace(0, 5, num_bins + 1)
        binned = np.digitize(z_filt, bin_edges)
        binned = np.abs(binned.flatten())

        seconds = int(np.floor(np.shape(binned)[0] / fs))
        data = binned[0:int(seconds * fs)]
        data = data.reshape(fs, seconds, order="F").copy()

        data = data.max(0).flatten()
        num_frames = int(np.floor(data.shape[0] / frame_size))
        data = data[0:(num_frames * frame_size)]

        data = data.reshape(frame_size, num_frames, order='F').copy()
        frame_data = np.sum(data, axis=0).flatten()

        frame_data = (frame_data - 18) * 3.07 # !!!
        frame_data[frame_data < 0] = 0

        frame_time = np.linspace(start_ts, end_ts, frame_data.shape[0])
        frame_time = np.expand_dims(frame_time, axis=1)
        frame_data = np.expand_dims(frame_data, axis=1)
        output = np.hstack((frame_time, frame_data))

        return output


    def load_user_acc(self, user_id: str) -> pd.DataFrame:
        """Load user motion data."""
    
        return self._load_user_file(
            user_id,
            fname="acceleration.txt",
            delim=" ",
            cols=["timestamp", "acc_x", "acc_y", "acc_z"])

    def load_user_labels(self, user_id: str) -> pd.DataFrame:
        """Load PSG labels."""
        
        return self._load_user_file(
            user_id,
            fname="labels.txt",
            delim=" ",
            cols=["timestamp", "label"])

    def load_user_steps(self, user_id: str) -> pd.DataFrame:
        """Load user step data."""
    
        return self._load_user_file(
            user_id,
            fname="steps.txt",
            cols=["timestamp", "num_steps"])

    def load_user_heartrate(self, user_id: str) -> pd.DataFrame:
        """Load heartrate samples."""
    
        return self._load_user_file(
            user_id,
            fname="heartrate.txt",
            cols=["timestamp", "bp"])

    def load_user_data(self, user_id: str) -> UserData:
        """Load raw samples for motion, heartrate and psg labels."""
    
        df_labels = self.load_user_labels(user_id)
        df_motion = self.load_user_acc(user_id)
        df_heartrate = self.load_user_heartrate(user_id)
    
        # find the intersection of the samples on the time dimension
        min_ts = max(
            df_labels.timestamp.min(),
            df_motion.timestamp.min(),
            df_heartrate.timestamp.min()
        )
        max_ts = min(
            df_labels.timestamp.max(),
            df_motion.timestamp.max(),
            df_heartrate.timestamp.max()
        )

        df_labels = df_labels[(df_labels.timestamp >= min_ts) & (df_labels.timestamp <= max_ts)]
        df_motion = df_motion[(df_motion.timestamp >= min_ts) & (df_motion.timestamp <= max_ts)]
        df_heartrate = df_heartrate[(df_heartrate.timestamp >= min_ts) & (df_heartrate.timestamp <= max_ts)]
        df_activity = pd.DataFrame(self._compute_activity_counts(df_motion),
                                   names=["timestamp", "activity_count"])

        return UserData(labels=df_labels,
                        motion=df_motion,
                        heartrate=df_heartrate,
                        activity_count=df_activity)

        
        
        
    
