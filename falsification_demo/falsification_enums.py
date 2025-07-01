"""Enums."""

from enum import Enum


class ENV_MUTATION_TYPE(Enum):
    ADD = 0
    REMOVE = 1
    REPLACE = 2
    
class ENV_MUTATION_BREADTH(Enum):
    RANDOM = 0
    CONSTANT = 1

class ENV_MUTATION_DEPTH(Enum):
    UNLIMITED = 0
    LOCAL_PERTUBATION = 1

class INCREMENTAL_SIM(Enum):
    NO = 0
    YES = 1

class NODE_SELECTION(Enum):
    RANDOM = 0
    RRT = 1
    RRT_SIMPLIFIED = 2