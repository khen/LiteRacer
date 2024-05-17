"""Run vehicle in specified environment until termination."""

import matplotlib.pyplot as plt
from resources.config import visualizer_config
from components.worldSimulator import WorldSimulator
from utils.enums import DrawObservation, VehicleStatus
import argparse


# read inline arguments
parser = argparse.ArgumentParser()
parser.add_argument('number_of_runs', nargs='?', type=int, default=1)
args = parser.parse_args()

# run the requested amount of times
for run in range(args.number_of_runs):
    # init world
    world_simulator = WorldSimulator()

    # visualize run
    if visualizer_config.visualization == DrawObservation.ON:
        world_simulator.open_world_view(visualizer_config.window_position_x,visualizer_config.window_position_y,visualizer_config.open_observation_view)

    # run
    print("Run #"+str(run+1)+": running...")
    states, observations, actions = world_simulator.vehicle.run()
        
    # check and report vehicle status
    if world_simulator.vehicle.status == VehicleStatus.FINISH:
        print("Goal reached.")
    elif world_simulator.vehicle.status == VehicleStatus.UNSAFE:
        print("Vehicle unsafe.")
    else:
        print("Vehicle timed-out.")

    # visualize end of run and block
    if visualizer_config.visualization == DrawObservation.ON:
        plt.show(block=True)  
    elif visualizer_config.visualization == DrawObservation.RUN_TERMINATION_ONLY:
        world_simulator.open_world_view(visualizer_config.window_position_x,visualizer_config.window_position_y,visualizer_config.open_observation_view)
        plt.show(block=True)

    world_simulator.kill() # close open figures and realease resources