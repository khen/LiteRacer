"""Falsification attempt using Bayesian optimization (3 obstacles)."""
### To be able to call files from parent folder
import os, sys
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
###

from bayes_opt import BayesianOptimization, acquisition
from meta_planning.controllableSimulation import ControllableSimulation
from LiteRacer.resources.config import env_config
from LiteRacer.components.vehicle import Vehicle


def simulation_function_wrapper(obs1_x, obs1_y, obs2_x, obs2_y, obs3_x, obs3_y):
    """Step function for Bayesian optimizer."""

    simulation = ControllableSimulation()

    radius = env_config.radius_of_random_obstacles
    simulation.add_obstacles_relative_to_track([[obs1_x,obs1_y,radius],[obs2_x,obs2_y,radius],[obs3_x,obs3_y,radius]])

    simulation.vehicle.run()
    simulation.vehicle.kill()
    
    output_signal = -1*simulation.distance_to_specification_failure()

    simulation.kill()

    return output_signal



# override env_config, to control the sampling of random obstacles through the Bayes optimizer
env_config.number_of_random_obstacles_at_init = 0

failure_counter = 0
total_control_loops = 0
total_envs = 0

for search_attempt_counter in range(1,3):

    print(f"Search attempt: {search_attempt_counter}")
    
    # initialize optimizer
    # bounded region of parameter space
    pbounds = {'obs1_x': (0, 1), 'obs1_y': (0, 1), 'obs2_x': (0, 1), 'obs2_y': (0, 1), 'obs3_x': (0, 1), 'obs3_y': (0, 1), }
    acq = acquisition.UpperConfidenceBound(kappa=2.5)
    optimizer = BayesianOptimization(
                    f=simulation_function_wrapper,
                    pbounds=pbounds,
                    acquisition_function=acq,
                    verbose=2)
    
    # optimization loop
    for i in range(1,701):
        next_point = optimizer.suggest()
        signal = simulation_function_wrapper(**next_point)
        optimizer.register(params=next_point, target=signal)
        print(f"Test #{i}, test score: {signal}.")

        if signal == 0:
            print(f"Failure found.")
            failure_counter = failure_counter + 1
            break
    
    if signal != 0:        
        print("Search timed-out.")
    print(f"Tested {i} envs using {Vehicle.control_loop_counter} total control loops.\n")

    total_envs = total_envs + i
    total_control_loops = total_control_loops + Vehicle.control_loop_counter
    Vehicle.control_loop_counter = 0
    
print(f"Overall: {total_envs} total envs tested, {total_control_loops} total control loops.")
print(f"Search success rate: {failure_counter}/{search_attempt_counter}.")