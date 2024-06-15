from enum import Enum
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

WearableFeatures = namedtuple(
    "WearableFeatures",
    [
        "motion", "heartrate", "cos_time", "real_time", "labels"
    ])

