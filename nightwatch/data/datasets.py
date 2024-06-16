import json
import random

from tqdm import tqdm
from pathlib import Path

import numpy as np

from nightwatch.data.common import (
    SleepStage,
    SleepDatasetReader,
    SleepFeatureExtractor
)
from nightwatch.data.features import SleepFeatureExtractor


class SleepSeqDataset:
    """A dataset class that reads sleep stage examples from HDF5 files.

    The file is expected to contain an array of samples. Each sample
    is a dictionary with the following keys:
    
    - user_id: a unique identifier for the user (str)
    - features: a dictionary containing arrays for motion, heartrate and time
    - label: an integer representing the sleep stage label (int)

    In order to create such files from raw datasets, use the
    `build_dataset` function.
    """

    def __init__(self, json_path: str):
        """Read samples from the given file.

        Args:
        json_path (str): path to the sample file.
        """
        if json_path is None:
            raise ValueError("json_path is None")
        
        self.file_path = Path(json_path)
        if not self.file_path.exists() or\
           not self.file_path.is_file():
            raise IOError(f"could not read {json_path}")

        self.feature_names = [
            "motion", "heartrate",
            "cos_time", "real_time"
        ]
        self.num_features = len(self.feature_names)
        self.num_labels = len(SleepStage)
        self.data = []
        self.load_data()

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieves the sample at the specified index."""
        return self.data[idx]

    def load_data(self):
        """
        Loads and parses the JSON file into the dataset.
        """
        with open(self.file_path, 'r') as f:
            raw_data = json.load(f)
            for item in raw_data:
                feats = np.column_stack(
                    [
                        np.array(item["features"][name])\
                        for name in self.feature_names

                    ])
                self.data.append((feats, int(item['label'])))

    def build_dataset(reader: SleepDatasetReader,
                      extractor: SleepFeatureExtractor,
                      target_dir: str = ".",
                      window_size_min: int = 10,
                      window_stride_min: int = 5,
                      train_test_split: float = 0.9):
        """Builds the dataset using the provided reader and feature
        extractor, and splits it into training and test sets.

        Args:
            reader (SleepDatasetReader): An instance of SleepDatasetReader
                used to read raw sleep data.
            extractor (SleepFeatureExtractor): An instance of
                SleepFeatureExtractor used to extract features from raw data.
            target_dir (str, optional): The directory where the processed
                dataset will be saved. Defaults to the current directory.
            window_size_min (int, optional): input size in minutes. Default: 10
            window_stride_min (int, optional): window stride in minutes. Default: 5
            train_test_split (float, optional): The ratio of training to
                testing data. Defaults to 0.9.

        Returns:
            None
        """
        if reader is None:
            raise ValueError("reader is None")
        if extractor is None:
            raise ValueError("extractor is None")
        if train_test_split < 0 or train_test_split > 1:
            raise ValueError("train_test_split must be a float between 0 and 1")

        if not target_dir:
            raise ValueError("target dir must contain a value")
        
        target_path = Path(target_dir)
        target_path.mkdir(exist_ok=True, parents=True)

        samples = []
        for user_id in tqdm(reader.get_users()):

            user_data = reader.get_user_data(user_id)
            features = extractor.compute_features(user_data)

            max_seq_len = min(
                features.motion.shape[0],
                features.heartrate.shape[0],
                features.cos_time.shape[0],
                features.real_time.shape[0],
                features.labels.shape[0]
            )

            # Data is sampled every 30s.
            window_size_samples = window_size_min * 2
            window_stride_samples = window_stride_min * 2

            for wend in range(window_size_samples,
                              max_seq_len - window_size_samples,
                              window_stride_samples):
                label = features.labels[wend + 1]

                wstart = wend - window_size_samples

                samples.append({
                    "user_id": user_id,
                    "features": {
                        "motion": list(features.motion[wstart:wend]),
                        "heartrate": list(features.heartrate[wstart:wend]),
                        "cos_time": list(features.cos_time[wstart:wend]),
                        "real_time": list(features.real_time[wstart:wend])
                    },
                    "label": int(label)
                })

        random.shuffle(samples)
        
        split_index = int(train_test_split * len(samples))
        
        with open(target_path / "train.json", "wt") as train_file:
            json.dump(
                samples[:split_index],
                train_file,
                indent=2
            )
        with open(target_path / "test.json", "wt") as test_file:
            json.dump(
                samples[split_index:],
                test_file,
                indent=2
            )
        
        
        
        
                      


