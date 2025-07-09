"""An extension of Simulation."""

from math import dist
import numpy as np
import random
import shapely.geometry as sg
from numpy.random import normal
from numpy import sin, cos, inf, pi
from LiteRacer.utils.enums import VehicleStatus
from LiteRacer.components.simulation import Simulation
from .enums import INCREMENTAL_SIM, ENV_MUTATION_TYPE, ENV_MUTATION_BREADTH, ENV_MUTATION_DEPTH, NODE_SELECTION
from . import config


class ControllableSimulation(Simulation): #TODO: move interface to separate class
    """An extension of Simulation, adding functinoality to serve as a meta-state in meta-planning."""

    control_loop_saved_counter = 0
    
    def apply_random_meta_control(self):
        """Apply meta-control."""

        # mutate obstacle collection and let vehicle run
        simulation_copy = self.copy()

        # find out how many obstacles to remove
        if config.env_mutation_breadth == ENV_MUTATION_BREADTH.RANDOM:
            mutation_breadth = random.randint(1,len(simulation_copy.obstacles_in_WF))
        elif config.env_mutation_breadth == ENV_MUTATION_BREADTH.CONSTANT:
            mutation_breadth = config.number_of_obstacles_to_mutate

        # remove old obstacles
        history_compromise_timestamp = inf
        obstacles_removed_in_WF = []
        if config.env_mutation_type == ENV_MUTATION_TYPE.REMOVE \
            or config.env_mutation_type == ENV_MUTATION_TYPE.REPLACE:
            obstacles_removed_in_WF, removed_history_compromise_timestamp = simulation_copy._remove_obstacles_from_env_and_calc_HCTS(mutation_breadth)
            history_compromise_timestamp = min([removed_history_compromise_timestamp,history_compromise_timestamp])

        # add new obstacles
        obstacles_added = []
        if config.env_mutation_type == ENV_MUTATION_TYPE.ADD \
            or config.env_mutation_type == ENV_MUTATION_TYPE.REPLACE:
            # limited depth replacement
            if config.env_mutation_depth == ENV_MUTATION_DEPTH.LOCAL_PERTUBATION:
                obstacles_to_add_in_WF = []
                for obstacle in obstacles_removed_in_WF:
                    # sample a valid obstacle location
                    new_obstacle_is_valid = False
                    while not new_obstacle_is_valid:
                        new_x = normal(loc=obstacle[0], scale=config.obs_perturbation_stdv)
                        new_y = normal(loc=obstacle[1], scale=config.obs_perturbation_stdv)
                        if simulation_copy._track_shape_trimmed.contains(sg.Point([new_x,new_y])):
                            new_obstacle_is_valid = True
                    obstacles_to_add_in_WF.append([new_x,new_y,obstacle[2]])
                
                obstacles_added, added_history_compromise_timestamp = simulation_copy._add_obstacles_to_env_and_calc_HCTS(mutation_breadth, obstacles=obstacles_to_add_in_WF)
            # unlimited depth replacement
            else:
                obstacles_added, added_history_compromise_timestamp = simulation_copy._add_obstacles_to_env_and_calc_HCTS(mutation_breadth)
                
            history_compromise_timestamp = min([added_history_compromise_timestamp,history_compromise_timestamp])
        

        # reset vehicle to past state before restarting the simulation
        if config.incremental_simulation == INCREMENTAL_SIM.NO:
            simulation_copy.rollback(to_timestamp=0)
            control_steps_saved = 0
        elif config.incremental_simulation == INCREMENTAL_SIM.YES:
            simulation_copy.rollback(to_timestamp=(history_compromise_timestamp-1))
            control_steps_saved = history_compromise_timestamp

        # run vehicle simulation
        simulation_copy.vehicle.run() 
        simulation_control = [obstacles_removed_in_WF,obstacles_added]

        ControllableSimulation.control_loop_saved_counter = ControllableSimulation.control_loop_saved_counter + control_steps_saved

        return simulation_copy, simulation_control # simulation_control, states, observations, actions
    
   
    def _add_obstacles_to_env_and_calc_HCTS(self, suggested_number_of_obstacles, obstacles=None):
        """Adds the suggested number of obstacles to the env (at random), or, if given, the specified obstacle list. """

        if obstacles is not None:
            # add the given obstacles
            self.add_obstacles_in_WF(obstacles, verify_intersection_with_vehicle=False)
            obstacles_added = obstacles
        else:
            # no obstacles given, add random obstacles instead
            # make sure to not pass the upper limit of obstacles
            num_of_obs_to_add = min([max([0, config.max_number_of_obstacles-len(self.obstacles_in_WF)]),\
                                     suggested_number_of_obstacles])
            obstacles_added = self.add_random_obstacles(num_of_obs_to_add, verify_intersection_with_vehicle=False)

        history_compromise_timestamp = len(self._vehicle_state_history)
        for obstacle in obstacles_added:
            timestamp = self.calc_history_compromise_timestamp(obstacle)
            if timestamp < history_compromise_timestamp:
                history_compromise_timestamp = timestamp
        
        return obstacles_added, history_compromise_timestamp
    

    def _remove_obstacles_from_env_and_calc_HCTS(self, suggested_number_of_obstacles, obstacle_indices=None):
        """Removes the suggested number of obstacles from the env (at random), or, if given, the specified obstacle list. """

        if obstacle_indices is not None:
            # remove the given obstacles
            obstacles_removed = self.remove_obstacles(obstacle_indices)
        else:
            # no obstacles given, remove random obstacles instead
            # make sure to not pass the lower limit of obstacles
            num_of_obs_to_remove = min([len(self.obstacles_in_WF), suggested_number_of_obstacles])
            obstacles_removed = self.remove_random_obstacles(num_of_obs_to_remove)

        history_compromise_timestamp = len(self._vehicle_state_history)
        for obstacle in obstacles_removed:
            timestamp = self.calc_history_compromise_timestamp(obstacle)
            if timestamp < history_compromise_timestamp:
                history_compromise_timestamp = timestamp
        
        return obstacles_removed, history_compromise_timestamp


    def calc_history_compromise_timestamp(self, changed_obstacle):
        """Finds the timestamp in which the vehicle should have seen the given obstacle, had it been there originally."""

        changed_obstacle_circle = sg.Point(changed_obstacle[0],changed_obstacle[1]).buffer(changed_obstacle[2])
        timestamp = 0
        for observation in self._observation_wedges_history:
            if not observation.intersection(changed_obstacle_circle).is_empty:
                break
            else:
                timestamp = timestamp + 1
        if timestamp >= len(self._observation_wedges_history): # no compromise
            return inf
        else:
            return timestamp


    def is_specification_failure(self): 
        """Checks if the vehicle failed the specification (this is the STATUS predicate, as described in the paper)."""

        return self.vehicle.status == VehicleStatus.UNSAFE  # here the specification is the failure of the vehicle


    def distance_to_specification_failure(self):
        """Distance to failing the specification."""

        if self.is_specification_failure(): 
            return 0
        else:
            return self.get_distance_to_collision_throughout_history(only_in_sensing_range=False)


    def distance_between_simulations(self, other_simulation): #### TODO: IMPLEMENT in parent class
        """Return distance between simulations, accounting for both env and trajectory."""

        # measure area between the two trajectory curves
        # get x and y vectors
        x = [state[0] for state in self._vehicle_state_history]
        y = [state[1]+2 for state in self._vehicle_state_history]

        other_x = [state[0] for state in other_simulation._vehicle_state_history]
        other_y = [state[1]+2 for state in other_simulation._vehicle_state_history]

        evened_other_y = np.interp(x, other_x, other_y)

        max_y = np.maximum.reduce([y,evened_other_y])
        min_y = np.minimum.reduce([y,evened_other_y])

        # area up until top border - area up until bottom border
        area1 = np.trapz(max_y, x)
        area2 = np.trapz(min_y, x)
        area = area1-area2

        env_dist = self.distance_between_envs(other_simulation)

        return env_dist+area


    def distance_between_envs(self, other_simulation):
        """Return an estimated distance between envs."""
        
        # distance estimated as explained in the paper
        N = 32 # number of points to sample for distance estimation
        
        # distance from this simulation to other simulation
        sum1 = 0
        for i in range(N):
            p_self = self._get_random_obstructed_point()
            
            dist_p_to_simulation_obstacles = [max([0,dist(obs[0:2],p_self)-obs[2]]) for obs in other_simulation.obstacles_in_WF]
            sum1 =  sum1 + min(dist_p_to_simulation_obstacles)
        avg1 = sum1/N
        
        # distance from other simulation to this simulation
        sum2 = 0
        for i in range(N):
            p_simulation = other_simulation._get_random_obstructed_point()
            
            dist_p_to_self_obstacles = [max([0,dist(obs[0:2],p_simulation)-obs[2]]) for obs in self.obstacles_in_WF]
            sum2 =  sum2 + min(dist_p_to_self_obstacles)
        avg2 = sum2/N

        #print("dists: "+str(avg1)+", "+str(avg2))
        # return symmetric distance
        return (avg1+avg2)/2          *3 # number of obstacles   ###
    

    def _get_random_obstructed_point(self):
        """Return a random point from within one of the obstacles."""

        random_obs = self.obstacles_in_WF[random.randint(0,len(self.obstacles_in_WF)-1)]
        random_angle = random.random()*2*pi
        random_r = random.random()*random_obs[2]
        random_point = [random_obs[0]+cos(random_angle)*random_r, random_obs[1]+sin(random_angle)*random_r]
        
        return random_point