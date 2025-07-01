"""Falsification attempt using a genetic algorithm for environement search."""
### To be able to call files from parent folder
import os, sys
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
###

import random
from falsification_demo.controllableSimulation import ControllableSimulation
from components.vehicle import Vehicle
from resources.config import env_config


failure_counter = 0
total_control_loops = 0
total_envs = 0

for search_attempt_counter in range(1,3):
    failure_case_found = False
    print(f"Search attempt: {search_attempt_counter}")

# override env_config, to control the obstacles through the genetic algorithm
    env_config.number_of_random_obstacles_at_init = 3
    # sample first generation: 4 simulations
    simulation1 = ControllableSimulation()
    simulation2 = ControllableSimulation()
    simulation3 = ControllableSimulation()
    simulation4 = ControllableSimulation()
    # run the simulation in each simulation
    simulation1.vehicle.run()
    simulation2.vehicle.run()
    simulation3.vehicle.run()
    simulation4.vehicle.run()
    env_config.number_of_random_obstacles_at_init = 0

    generation = 1
    while generation <= 200:
        print(f"Generation: {generation}")

        # calc distance to collistion scores
        score1 = simulation1.distance_to_specification_failure()
        score2 = simulation2.distance_to_specification_failure()
        score3 = simulation3.distance_to_specification_failure()
        score4 = simulation4.distance_to_specification_failure()
        
        sum_of_scores = score1+score2+score3+score4
        norm_score1 = (sum_of_scores-score1)/sum_of_scores
        norm_score2 = (sum_of_scores-score2)/sum_of_scores
        norm_score3 = (sum_of_scores-score3)/sum_of_scores
        norm_score4 = (sum_of_scores-score4)/sum_of_scores

        # print(f"scores: {score1},{score2},{score3},{score4}")

        if score1 == 0 or score2 == 0 or score3 == 0 or score4 == 0:
            # solution found, stop the process
            failure_case_found = True
            break

        # else - create new generation
        generation = generation + 1

        # based on the DTF, sample two simulations and add a random mutation to each one, to create two "single parent" children
        selected_simulations = random.choices(population=[simulation1,simulation2,simulation3,simulation4],weights=[norm_score1,norm_score2,norm_score3,norm_score4],k=6)

        mutation_simulation1, mutation = selected_simulations[0].apply_random_meta_control()
        mutation_simulation2, mutation = selected_simulations[1].apply_random_meta_control()

        # based on the DTF, sample two pairs of simulations and merge each pair's obstacle set randomly, to create two "two parent" children
        crossover_simulation1 = ControllableSimulation()
        selected_obs = random.sample(selected_simulations[2].obstacles_in_WF + selected_simulations[3].obstacles_in_WF, 3)
        crossover_simulation1.add_obstacles_in_WF(selected_obs)
        crossover_simulation1.vehicle.run()

        crossover_simulation2 = ControllableSimulation()
        selected_obs = random.sample(selected_simulations[4].obstacles_in_WF + selected_simulations[5].obstacles_in_WF, 3)
        crossover_simulation2.add_obstacles_in_WF(selected_obs)
        crossover_simulation2.vehicle.run()

        # update generation
        simulation1.kill()
        simulation2.kill()
        simulation3.kill()
        simulation4.kill()

        simulation1 = mutation_simulation1
        simulation2 = mutation_simulation2
        simulation3 = crossover_simulation1
        simulation4 = crossover_simulation2
    
    if failure_case_found:
        failure_counter = failure_counter+1
        print(f"Failure found.")
    else:
        print("Search timed-out.")
    print(f"Tested {generation*4} envs using {Vehicle.control_loop_counter} control loops.\n")

    simulation1.kill()
    simulation2.kill()
    simulation3.kill()
    simulation4.kill()

    total_envs = total_envs + generation*4
    total_control_loops = total_control_loops + Vehicle.control_loop_counter
    Vehicle.control_loop_counter = 0


print(f"Overall: {total_envs} total envs tested, {total_control_loops} total control loops.")
print(f"Search success rate: {failure_counter}/{search_attempt_counter}.")
