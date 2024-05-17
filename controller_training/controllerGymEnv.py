"""An implementation of gym.Env. Used to define the NN input/output format and reward function for training.
Used for training (using stable_baselines3) and later for inference with the NN."""

import gym
import numpy as np
from PIL import Image
from resources.config import env_config, vehicle_config
from training_util_funcs import collision_distance_score, goal_distance_score, calc_reward
import training_config
from utils.enums import VehicleStatus
from components.worldSimulator import WorldSimulator

class ControllerGymEnv(gym.Env):

    def __init__(self, verbose=0):
        self.world_simulator = None
        
        self.verbose = verbose

        self.action_space = gym.spaces.Box(
            low=np.array([vehicle_config.forward_speed_range[0], vehicle_config.steering_speed_range[0]]),
            high=np.array([vehicle_config.forward_speed_range[1], vehicle_config.steering_speed_range[1]]))
        
        N_CHANNELS = 1
        self.observation_scan_shape = (N_CHANNELS, vehicle_config.observation_image_size[0], vehicle_config.observation_image_size[1])
        self.observation_space = gym.spaces.Dict({
            "scan":
            gym.spaces.Box(low=0,
                           high=255,
                           shape=self.observation_scan_shape,
                           dtype=np.uint8),
            "steering":
            gym.spaces.Box(low=vehicle_config.steering_angle_range[0],
                           high=vehicle_config.steering_angle_range[1],
                           shape=(1,),
                           dtype=np.float32)
        })


    def step(self, action):

        self.step_counter += 1
        if self.verbose >= 2:
            print("Training step #"+str(self.step_counter))

        # execute given action
        self.world_simulator.vehicle.execute_control_action(action, vehicle_config.control_duration)


        ############## THIS PORTION IS LEFT FOR REFERENCE, REPLACE REWARD CALCULATION WITH YOUR OWN...

        # calc reward (based on diff between state values)
        post_goal_distance_score = goal_distance_score(self.world_simulator)
        post_collision_distance_score = collision_distance_score(self.world_simulator)

        reward, self.prev_goal_distance_score, self.prev_collision_distance_score = \
            calc_reward(self.world_simulator.vehicle.status,
                        self.prev_goal_distance_score,
                        post_goal_distance_score,
                        self.prev_collision_distance_score,
                        post_collision_distance_score,
                        self.world_simulator)
                        # world_simulator is only provided to provide access to additional information on vhicle status for some advanced reward options
        
        self.prev_goal_distance_score = post_goal_distance_score
        self.prev_collision_distance_score = post_collision_distance_score

        ##########################################

        # calc doneness
        if self.world_simulator.vehicle.status == VehicleStatus.UNSAFE or \
            self.world_simulator.vehicle.status == VehicleStatus.FINISH or \
            self.step_counter > training_config.max_training_steps:
            done = True
        else:
            done = False

        # calc posterior observation
        obs = self._get_current_observation_for_controller()

        return (obs, reward, done, {})


    def reset(self):
        if self.verbose >= 1:
            print("Training reset")

        if self.world_simulator is not None:
            self.world_simulator.kill()

        self.world_simulator = WorldSimulator()

        self.prev_goal_distance_score = goal_distance_score(self.world_simulator)
        self.prev_collision_distance_score = collision_distance_score(self.world_simulator)

        self.step_counter = 0
        return self._get_current_observation_for_controller()
    

    def _get_current_observation_for_controller(self):
        # get steering from state
        steering = self.world_simulator.vehicle.get_state()[3]
        # get a sensor image
        if self.world_simulator.vehicle.status == VehicleStatus.SAFE:
            observation_image = self.world_simulator.vehicle.sensor.get_observation_image()
        else:
            observation_image = Image.new('L', [self.observation_scan_shape[2], self.observation_scan_shape[1]])
        # format inputs and return
        return self.world_simulator.vehicle.controller.arrange_controller_input(observation_image, steering)
    
