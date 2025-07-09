"""Vehicle intrinsic parameters."""

from math import pi, atan2
from ...components.controller import Controller
from ...components.sensor import Sensor
from ...utils.enums import SensingFreq
from .env_config import track_drv_func


### vehicle config
width = 0.2
length = 0.4
initial_state = [0.01, 0.01, atan2(track_drv_func(0.01),1), 0] # format: [x, y, theta, phi (steering angle)]
forward_speed_range = [0.05, 0.4]
steering_speed_range = [-pi/180*10, pi/180*10]
steering_angle_range = [-pi/180*60, pi/180*60]

### controller config
controllerClass = Controller # do not change unless using a custom controller that overrides "set_controller_model"
controller_model_path = 'resources/controller_models/3b' # may use None for training mode (will not load a controller)
controller_model_type = 'SAC' # stable_baselines model type. may use None for training mode (will not load a controller) or if using a custom model
control_duration = 1 # (sec.) detemines the frequency of choosing new control actions
delta_t = 0.25 # (sec.) time discretization resolution for continuous motion simulation.
               # reduce value to increase smoothness (vs. computation speed). use same value as control_duration to ignore
run_timeout = 10000 # maximal amount of control steps per run

### sensor config
sensing_freq = SensingFreq.CONTINUOUS #SensingFreq.ON_REQUEST # the frequency in which sensing is invoked
sensorClass = Sensor # do not change unless overriding functionality
sensor_angle_range = [-2*pi/5, 2*pi/5]
sensor_max_range = 2
sensor_dpi = 100 # base value is 100, increase to increase sensor "sensitivity" (and observation visulizer resolution)
observation_image_size = [100, 50] # downsampled resolution of the observation image (input to controller).
                                   # original size is [2*sensor_max_range, sensor_max_range]*sensor_dpi
## set relative poistion of sensor (frame) in vehicle (frame, whose origin is located in rear-center)
sensor_origin_offset_x = 0*length # use 0 to overlap with vehicle frame, use "length" to place sensor in the front-center
sensor_origin_offset_y = 0