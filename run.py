"""Run vehicle in specified environment until termination."""

import matplotlib.pyplot as plt
from resources.config import visualizer_config
from components.simulation import Simulation
from components.vehicle import Vehicle
from utils.enums import Visualization, VehicleStatus
import argparse


# read inline arguments
parser = argparse.ArgumentParser()
parser.add_argument('number_of_runs', nargs='?', type=int, default=1)
args = parser.parse_args()

# run the requested amount of times
for run in range(args.number_of_runs):
    # init simulation
    simulation = Simulation()

    # visualize run
    if visualizer_config.visualization == Visualization.ON_AND_BLOCK_ON_TERMINATION or \
         visualizer_config.visualization == Visualization.ON_AND_NO_BLOCK_ON_TERMINATION:
        simulation.open_visualizer(visualizer_config.window_position_x, visualizer_config.window_position_y, visualizer_config.open_observation_view)

    # run
    print(f"Run #{run+1}: running...")
    states, observations, actions = simulation.vehicle.run()
        
    # check and report vehicle status
    if simulation.vehicle.status == VehicleStatus.FINISH:
        print("Goal reached.\n")
    elif simulation.vehicle.status == VehicleStatus.UNSAFE:
        print("Vehicle unsafe.\n")
    else:
        print("Vehicle timed-out.\n")

    # visualize end of run and block
    if visualizer_config.visualization == Visualization.ON_AND_BLOCK_ON_TERMINATION:
        plt.show(block=True)
    if visualizer_config.visualization == Visualization.TERMINATION_ONLY:
        simulation.open_visualizer(visualizer_config.window_position_x, visualizer_config.window_position_y, visualizer_config.open_observation_view)
        plt.show(block=True)

    simulation.kill() # close open figures and realease resources

    print(f"Control loops: {Vehicle.control_loop_counter}.")
    Vehicle.control_loop_counter = 0
    