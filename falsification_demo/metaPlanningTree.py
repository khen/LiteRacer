"""Meta-planning tree."""

import numpy as np
from falsification_demo.controllableSimulation import ControllableSimulation
from falsification_enums import NODE_SELECTION
import falsification_config

class MetaPlanningTree:

    class Node:
        def __init__(self, meta_state):
            self.meta_state            = meta_state
            self.parent_node           = None
            self.incoming_meta_control = None

            self._distance_to_goal_condition = None


    def __init__(self, initial_meta_state):
        self.initial_node   = self.Node(initial_meta_state)
        self.max_steps = falsification_config.max_steps
        self.node_selection = falsification_config.node_selection
        self.goal_bias_rate = falsification_config.goal_bias_rate

        self.number_of_simulated_samples = 0
        self.node_list = [] # TODO: CAN REPLACE THIS WITH A QUEUE

    def kill(self):
        for node in self.node_list:
            node.meta_state.kill()

    def get_tree_size(self):
        return len(self.node_list)

    def plan(self):
        """Build a forward planning tree from start to goal meta-states."""

        self.node_list = [self.initial_node]
        i=0
        while i < self.max_steps:
            print('Meta-step: '+str(i+1))

            # select node
            selected_node, sampled_meta_state = self._select_node()

            # when using RRT for selection, verify that the sampled meta-state isn't a goal meta-state, otherwise _expand_node selected node
            if (sampled_meta_state is not None) and sampled_meta_state.is_specification_failure() == True:
                print("Sampled goal meta-state during RRT node selection!")
                return sampled_meta_state
            else:
                new_nodes, goal_reached = self._expand_node(selected_node)
                if goal_reached:
                    print("Goal meta-state reached!")
                    return self.node_list[-1].meta_state
            
            i=i+1
        
        print("Meta-planning timed out!")
        return None # cannot find path


    def _select_node(self):
        """Select a node to expand based on different strategies"""

        sampled_meta_state = None

        if self.node_selection == NODE_SELECTION.RANDOM:
            if np.random.rand() > self.goal_bias_rate:
                # select node at random
                node = self.node_list[np.random.randint(0,len(self.node_list))]
            else:  
                # select node closest to goal region
                node = self._get_nearest_node_to_goal_condition()

        elif self.node_selection == NODE_SELECTION.RRT or self.node_selection == NODE_SELECTION.RRT_SIMPLIFIED:
            if np.random.rand() > self.goal_bias_rate:
                # select node closest to a random state
                sampled_meta_state = self._get_random_meta_state()
                node = self._get_nearest_node(sampled_meta_state)
            else:  
                # select node closest to goal region
                node = self._get_nearest_node_to_goal_condition()

        return node, sampled_meta_state # for RRT cases, can also consider returning the sampled state, if I want to steer to it


    def _expand_node(self, from_node, number_of_children=1):
        """Expand a tree node by applying a random meta-control."""

        # TODO: support exapnsion of multiple children at once
        
        # apply meta control
        posterior_meta_state, meta_action = from_node.meta_state.apply_random_meta_control()
        posterior_meta_state.kill() # stop the simulation to release resources (as not planning to continue it later)

        # create node
        new_node = self.Node(posterior_meta_state)
        new_node.parent_action = meta_action
        new_node.parent = from_node
        new_node._distance_to_goal_condition = posterior_meta_state.distance_to_specification_failure()
        self.node_list.append(new_node)

        # check if goal reached # TODO: check why the distance to goal is not always zero when vehicle is unsafe
        if new_node.meta_state.is_specification_failure():
            goal_reached = True
        else:
            goal_reached = False

        return new_node, goal_reached
    

    def _get_random_meta_state(self):
        """Sample random meta-state."""

        # create a new simulation with random obstacles (number of obstacles is assumed to be constant here)
        rand_simulation = ControllableSimulation()

        # run simulation
        if self.node_selection != NODE_SELECTION.RRT_SIMPLIFIED:
            rand_simulation.vehicle.run()
            rand_simulation.kill() # no need to resume this simulation
            self.number_of_simulated_samples = self.number_of_simulated_samples + 1

        return rand_simulation


    def _get_nearest_node(self, to_meta_state):
        """Find the nearest node in node_list to a given meta-state."""

        dlist = [self._distance_between_meta_states(to_meta_state,node.meta_state) for node in self.node_list]
        minind = dlist.index(min(dlist))

        return self.node_list[minind]
    

    def _get_nearest_node_to_goal_condition(self):
        """Find the nearest node in node_list to the goal."""

        dlist = [self._distance_to_goal_condition(node) for node in self.node_list]
        minind = dlist.index(min(dlist))

        return self.node_list[minind]
    

    def _distance_to_goal_condition(self, node):
        """Return the distance from a given node to the goal."""

        # if hasn't been calculated (and cached) before
        if node._distance_to_goal_condition is None:
            node._distance_to_goal_condition = node.meta_state.distance_to_specification_failure()
        return node._distance_to_goal_condition


    def _distance_between_meta_states(self, meta_state1, meta_state2):
        """Return the distance between meta_state1 and meta_state2."""
        
        if self.node_selection == NODE_SELECTION.RRT_SIMPLIFIED:
            return meta_state1.distance_between_envs(meta_state2)
        else:
            return meta_state1.distance_between_simulations(meta_state2)