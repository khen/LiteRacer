"""Enums."""

from enum import Enum


class Visualization(Enum):
    OFF = 0
    ON = 1
    TERMINATION_STATE_ONLY = 2

class DrawObservationInVisualizer(Enum):
    OFF = 0
    CURRENT = 1
    CURRENT_AND_HISTORY = 2

class VehicleStatus(Enum):
    UNSAFE = 0
    SAFE = 1
    FINISH = 2
    UNKNOWN = 3

class SensingFreq(Enum):
    ON_REQUEST = 0 # update sensor image only as input to the controller, per contoller's request
    CONTINUOUS = 1  # update sensor image on vehicle movement