"""Enums."""

from enum import Enum


class DrawObservation(Enum):
    OFF = 0
    RUN_TERMINATION_ONLY = 1
    ON = 2


class DrawObservation(Enum):
    OFF = 0
    ON = 1
    ON_INCL_HISTORY = 2

class VehicleStatus(Enum):
    UNSAFE = 0
    SAFE = 1
    FINISH = 2
    UNKNOWN = 3