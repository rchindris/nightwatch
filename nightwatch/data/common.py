from enum import Enum
from abc import ABC, abstractmethod
from typing import List
from collections import namedtuple


class SleepStage(Enum):
    wake = 0
    n1 = 1
    n2 = 2
    n3 = 3
    n4 = 4
    rem = 5
    unscored = -1

UserData = namedtuple(
    "UserData",
    [
        "psg", "motion", "heartrate", "activity_count"
    ])

UserFeatures = namedtuple(
    "UserFeatures",
    [
        "motion", "heartrate", "cos_time", "real_time", "labels"
    ])


class SleepDatasetReader(ABC):
    """Define the interface for a dataset."""

    @abstractmethod
    def get_users(self) -> List[str]:
        """Return a list of user IDs."""
        ...

    @abstractmethod
    def get_user_data(self) -> UserData:
        """Return raw user sleep data."""
        ...

class SleepFeatureExtractor(ABC):
    """Base class for feature extractors."""

    @abstractmethod
    def compute_features(raw_sample: UserData) -> UserFeatures:
        """Compute features from user data."""
        ...
