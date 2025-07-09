"""Vehicle."""

from cmath import inf
from math import sin, cos, tan, pi
from shapely import affinity
import shapely.geometry as sg
from ..utils.enums import VehicleStatus, SensingFreq


class Vehicle():
    """Controllable vehicle modeled as kinematic bicycle and equipped with a sensor."""

    control_loop_counter = 0
    """Number of conrol loops invoked by Vehicles thus far."""

    VF_sensor_origin = None
    """Position of sensor frame relative to vehicle frame (located in rear-center)."""

    VF_shape = None
    """Shape of vehicle (static to avoid recalculation)."""

    def __init__(self, simulation, vehicle_config, initial_state=None):
        """Initiate a new Vehicle in the given Simulation according to given configuration."""

        self._state = initial_state
        self._state_listeners = []
        self.status = VehicleStatus.UNKNOWN

        # add sensor
        self.sensor = vehicle_config.sensorClass(simulation, vehicle_config)
        if vehicle_config.sensing_freq == SensingFreq.CONTINUOUS:
            self.add_listener_state(self.sensor.sense)

        # add controller
        self.controller = vehicle_config.controllerClass(vehicle_config)

        # create vehicle shape
        if Vehicle.VF_shape is None:
            Vehicle.width = vehicle_config.width
            Vehicle.length = vehicle_config.length
            Vehicle.VF_shape = Vehicle._calc_VF_vehicle_shape()
            
        # set sensor frame of reference
        if Vehicle.VF_sensor_origin is None:
            Vehicle.sensor_origin_offset_x = vehicle_config.sensor_origin_offset_x
            Vehicle.sensor_origin_offset_y = vehicle_config.sensor_origin_offset_y
            Vehicle.VF_sensor_origin = sg.Point([Vehicle.sensor_origin_offset_x,Vehicle.sensor_origin_offset_y])

        Vehicle.run_timeout = vehicle_config.run_timeout
        Vehicle.control_duration = vehicle_config.control_duration
        Vehicle.delta_t = vehicle_config.delta_t

    def get_state(self):
        """Get vehicle state."""

        return self._state

    def set_state(self, new_state):
        """Set vehicle state and invoke listeners."""

        if self._state != new_state:
            self._state = new_state
            for listener in self._state_listeners:
                listener()

    def add_listener_state(self, func, override_priority=False):
        """Add listener to state changes."""

        if override_priority:
            self._state_listeners.insert(0, func)
        else:
            self._state_listeners.append(func)

    def kill(self):
        """Release resources."""

        self._state_listeners = []
        self.sensor.kill()

    def run(self, number_of_control_steps=None):
        """Run vehicle for a given amout of control loops."""

        if number_of_control_steps is None or number_of_control_steps == inf:
            number_of_control_steps = Vehicle.run_timeout

        states = []
        observations = []
        actions = []

        for i in range(number_of_control_steps):
            # print("running control...")

            # get current state
            state = self.get_state()
            states.append(state)

            # get observation
            steering = state[3]
            observation_image = self.sensor.get_observation_image()
            observations.append(observation_image)
            
            # get control action
            action = self.controller.get_control_action(observation_image, steering)
            actions.append(action)
            # print(str(i)+": v="+str(action[0])+", steering_rate="+str(action[1]))

            # perform action (as long as safety is maintained)
            status = self.execute_control_action(action, Vehicle.control_duration)[0]
            if status == VehicleStatus.FINISH or status == VehicleStatus.UNSAFE:
                break

        state = self.get_state()
        states.append(state)
        
        Vehicle.control_loop_counter = Vehicle.control_loop_counter + i

        return states, observations, actions

    def execute_control_action(self, action, duration):
        """Execute a control action [v, steering rate] for "duration" seconds (in discrete "delta_t" increments), as long as safety status is maintained."""

        time_left = duration
        while time_left > Vehicle.delta_t:
            if self.status != VehicleStatus.SAFE: # verify status before each increment
                return self.status, duration-time_left
            self._execute_control_action_increment(action, Vehicle.delta_t)
            time_left = time_left-Vehicle.delta_t
        if time_left > 0:
            self._execute_control_action_increment(action, time_left)
            time_left = 0

        return self.status, duration-time_left # verify status before at the last increment


    def _execute_control_action_increment(self, action, delta_t):
        """Update vehicle state according to a discretized bicycle model."""

        v          = action[0]
        d_steering = action[1] # steering rate
        
        steering = self._state[3] + delta_t*d_steering
        if steering > self.controller.steering_angle_range[1]:
            steering = self.controller.steering_angle_range[1]
        elif steering < self.controller.steering_angle_range[0]:
            steering = self.controller.steering_angle_range[0]
        
        d_theta = v*tan(steering)/Vehicle.length
        theta = (self._state[2] + delta_t*d_theta) % (2*pi)

        d_y = v*sin(theta)
        y = self._state[1]+delta_t*d_y

        d_x = v*cos(theta)
        x = self._state[0]+delta_t*d_x

        self.set_state([x, y, theta, steering])
        
    @staticmethod
    def VF_to_SF(VF_shape):
        """Transform a shape from vehicle frame to sensor frame."""

        return affinity.translate(VF_shape, -Vehicle.sensor_origin_offset_x, -Vehicle.sensor_origin_offset_y)

    @staticmethod
    def SF_to_VF(SF_shape):
        """Transform a shape from sensor frame to vehicle frame."""
        
        return affinity.translate(SF_shape, Vehicle.sensor_origin_offset_x, Vehicle.sensor_origin_offset_y)

    @staticmethod
    def _calc_VF_vehicle_shape():
        """Return vehicle shape in vehicle frame based on its dimensions.
        
        Vehicle frame origin is located in the center of the rear bumper line.
        """

        return sg.Polygon([(0,-Vehicle.width/2),(0,Vehicle.width/2),(Vehicle.length,Vehicle.width/2),(Vehicle.length,-Vehicle.width/2)])