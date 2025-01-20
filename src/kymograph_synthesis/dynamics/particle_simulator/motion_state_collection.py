from enum import Enum, auto

# TODO: change back to string
class MotionStateCollection(Enum):
    ANTEROGRADE = "anterograde"
    STATIONARY = "stationary"
    RETROGRADE = "retrograde"

STATE_INDEX_MAPPING = {
    MotionStateCollection.RETROGRADE: 0,
    MotionStateCollection.STATIONARY: 1,
    MotionStateCollection.ANTEROGRADE: 2
}