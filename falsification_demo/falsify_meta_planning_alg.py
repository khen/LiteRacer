"""Falsification attempt using meta-planning."""
### To be able to call files from parent folder
import os, sys
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
###

from falsification_demo.controllableSimulation import ControllableSimulation
from falsification_demo.metaPlanningTree import MetaPlanningTree
from utils.enums import VehicleStatus
from components.vehicle import Vehicle

failure_counter = 0
total_tree_nodes = 0
total_meta_state_samples = 0
total_control_loops = 0
total_control_loops_saved = 0

for search_attempt_counter in range(1,3):
    print(f"Search attempt: {search_attempt_counter}")

    # "sample" initial meta state
    simulation = ControllableSimulation()
    simulation.vehicle.run()
    simulation.kill() # no need to continue this simulation later

    if simulation.vehicle.status == VehicleStatus.UNSAFE: 
        # failure found on initial sample
        failure_counter = failure_counter+1
        tree_node_counter = 1
        meta_state_sample_counter = 0
        print(f"Sampled a failure case as the planning-tree root! ({1} env tested).")
    else:
        # search via meta-planning
        planner = MetaPlanningTree(simulation)       
        result = planner.plan()
        # search ended
        if result is not None:
            failure_counter = failure_counter+1
            print(f"Failure found.")
        else:
            print("Search timed-out.")
        tree_node_counter = planner.get_tree_size()
        meta_state_sample_counter = planner.number_of_simulated_samples
        planner.kill()

        print(f"{tree_node_counter+meta_state_sample_counter} envs tested ({tree_node_counter} tree nodes and {meta_state_sample_counter} simulated samples for node selection).")

    print(f"Used {Vehicle.control_loop_counter} control loops. Additional {ControllableSimulation.control_loop_saved_counter} loops saved by using incremental calculation.\n")
    
    # maintain statistics
    total_tree_nodes = total_tree_nodes + tree_node_counter
    total_meta_state_samples = total_meta_state_samples + meta_state_sample_counter
    total_control_loops = total_control_loops + Vehicle.control_loop_counter
    total_control_loops_saved = total_control_loops_saved + ControllableSimulation.control_loop_saved_counter
    Vehicle.control_loop_counter = 0
    ControllableSimulation.control_loop_saved_counter = 0

print(f"Overall: {total_tree_nodes+total_meta_state_samples} total envs tested ({total_meta_state_samples} of which samples for node selection), {total_control_loops} total control loops (additional {total_control_loops_saved} loops saved by using incremental calculation).")
print(f"Search success rate: {failure_counter}/{search_attempt_counter}.")