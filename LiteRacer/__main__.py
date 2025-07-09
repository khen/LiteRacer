"""Run vehicle in specified environment until termination."""

import matplotlib.pyplot as plt
from .components.simulation import Simulation
from .components.vehicle import Vehicle
from .utils.enums import Visualization, VehicleStatus
import argparse


# read inline arguments
parser = argparse.ArgumentParser()
parser.add_argument('--runs', nargs='?', type=int, default=1)
parser.add_argument('--vis', nargs='?', default='on')
args = parser.parse_args()

if args.vis == 'off':
    visualization = Visualization.OFF
elif args.vis == 'end':
    visualization = Visualization.END_STATE_ONLY
else:
    visualization = Visualization.ON

number_of_runs = args.runs


# run the requested amount of times
for run in range(number_of_runs):
    # init simulation
    simulation = Simulation()

    # visualize run
    if visualization == Visualization.ON:
        simulation.open_visualizer()

    # run
    print(f"Run #{run+1}: running...")
    simulation.vehicle.run()
        
    # check and report vehicle status
    if simulation.vehicle.status == VehicleStatus.FINISH:
        print("Goal reached.")
    elif simulation.vehicle.status == VehicleStatus.UNSAFE:
        print("Vehicle unsafe.")
    else:
        print("Vehicle timed-out.")
    print(f"Control loops: {Vehicle.control_loop_counter}.\n")

    # visualize end of run
    if visualization == Visualization.END_STATE_ONLY:
        simulation.open_visualizer()
        
    # block program after each vehicle run, until visualizer window manually closed
    if visualization != Visualization.OFF:
        plt.show(block=True)

    simulation.kill() # close open figures and realease resources
    Vehicle.control_loop_counter = 0