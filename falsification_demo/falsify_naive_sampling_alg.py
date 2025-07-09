"""Falsification attempt using naive sampling environment sampling."""
### To be able to call files from parent folder
import os, sys
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
###

import simulation_config
from LiteRacer.components.simulation import Simulation
from LiteRacer.components.vehicle import Vehicle
from LiteRacer.utils.enums import VehicleStatus

test_counter = 0
failure_counter = 0
total_control_loops = 0

while failure_counter < 10:
    test_counter = test_counter + 1

    # run vehicle
    simulation = Simulation(simulation_config)
    simulation.vehicle.run()
        
    # check vehicle status
    if simulation.vehicle.status == VehicleStatus.FINISH:
        print(f"Test #{test_counter}: Goal reached.")
    elif simulation.vehicle.status == VehicleStatus.UNSAFE:
        failure_counter = failure_counter + 1
        print(f"Test #{test_counter}: Vehicle unsafe.")
        print(f"Failures so far: {failure_counter}.")
    else:
        print(f"Test #{test_counter}: Vehicle timed-out.")
    
    print(f"Control loops: {Vehicle.control_loop_counter}.\n")
    total_control_loops = total_control_loops + Vehicle.control_loop_counter
    Vehicle.control_loop_counter = 0

    simulation.kill()

print(f"Fail rate: {failure_counter}/{test_counter}.")
print(f"Total control loops: {total_control_loops}.")