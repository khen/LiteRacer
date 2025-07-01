"""Enums."""

from enum import Enum


class Visualization(Enum):
    OFF = 0
    TERMINATION_ONLY = 1
    ON_AND_BLOCK_ON_TERMINATION = 2 # blocks program after each vehicle run, until visualizer window manually closed
    ON_AND_NO_BLOCK_ON_TERMINATION = 3

class DrawObservationInVisualizer(Enum):
    OFF = 0
    ON = 1
    ON_INCL_HISTORY = 2

class VehicleStatus(Enum):
    UNSAFE = 0
    SAFE = 1
    FINISH = 2
    UNKNOWN = 3

class SensingFreq(Enum):
    ON_REQUEST = 0 # update sensor image only as input to the controller, per contoller's request
    CONTINUOUS = 1  # update sensor image on vehicle movement