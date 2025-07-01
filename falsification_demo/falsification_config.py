"""Falsification algorithm parameters."""
### To be able to call files from parent folder
import os, sys
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
###

from resources.config import env_config
from falsification_enums import INCREMENTAL_SIM, ENV_MUTATION_TYPE, ENV_MUTATION_BREADTH, ENV_MUTATION_DEPTH, NODE_SELECTION

### meta-system parameters
incremental_simulation = INCREMENTAL_SIM.YES
env_mutation_type = ENV_MUTATION_TYPE.REPLACE
env_mutation_breadth = ENV_MUTATION_BREADTH.RANDOM
number_of_obstacles_to_mutate = 1 # only applicable when env_mutation_breadth is constant
env_mutation_depth = ENV_MUTATION_DEPTH.LOCAL_PERTUBATION
obs_perturbation_stdv = 2 # only applicable when mutation depth is set to perturbation
max_number_of_obstacles = env_config.number_of_random_obstacles_at_init

### meta-planning parameters
max_steps = 700
node_selection = NODE_SELECTION.RRT
node_expansion_breadth = 1
goal_bias_rate = 0.8