"""Simulation configuration."""

from types import SimpleNamespace
from math import sin, cos, atan2, pi
from scipy.special import ellipeinc
from literacer.components.controller import Controller
from literacer.components.sensor import Sensor
from literacer.utils.enums import SensingFreq, DrawObservationInVisualizer


"""Environment parameters."""
env_config = SimpleNamespace()
### track parameters: a padded area aroud a given y=f(x) curve # TODO: support any parametarized curve
env_config.X_track_range = [0, 5*pi]
env_config.track_padding = 0.8 # TODO: if road is very thick, boundary can twist and should be smoothed
env_config.track_percision = 0.02
env_config.start_zone_portion = 0.1 # portion of track from start line without (random) obstacles
env_config.finish_line_portion = 0.1 # portion of the track to be designated finish line area--MAKE SURE to provide an area big enough to contain the vehicle
## track curve definition (requires first and second derivative to calc padded area)
env_config.track_vert_scale = 0.8
env_config.track_func         = lambda x:  env_config.track_vert_scale*sin(x)
env_config.track_drv_func     = lambda x:  env_config.track_vert_scale*cos(x)
env_config.track_sec_drv_func = lambda x: -env_config.track_vert_scale*sin(x)
env_config.track_arc_length   = lambda x: ((1+env_config.track_vert_scale**2)**0.5) * ellipeinc(x, (1+env_config.track_vert_scale**2)**(-0.5))

### obstacle parametes
# list of circular obstacles in format [c_x,c_y,r] # TODO: support obstacles of other shapes
    # c_x/c_y are to be specified a fraction in the range [0,1] 
    # c_x is the position along the the trajectory curve (whose x range is X_track_range), excluding the state_zone/finish_line portions.
    # c_y is the position between the long lower boundary and the upper boundary (on the axis perpendicular to the track curve on given x)
env_config.predefined_obstacles = [[0.1, 0.3, 0.43], [0.8, 0.8 , 0.25]]
env_config.number_of_random_obstacles_at_init = 3 # number of obstacles to add to env on initialization
env_config.radius_of_random_obstacles = 0.1


"""Vehicle intrinsic parameters."""
vehicle_config = SimpleNamespace()
### vehicle config
vehicle_config.width = 0.2
vehicle_config.length = 0.4
vehicle_config.initial_state = [0.01, 0.01, atan2(env_config.track_drv_func(0.01),1), 0] # format: [x, y, theta, phi (steering angle)]
vehicle_config.forward_speed_range = [0.05, 0.4]
vehicle_config.steering_speed_range = [-pi/180*10, pi/180*10]
vehicle_config.steering_angle_range = [-pi/180*60, pi/180*60]

### controller config
vehicle_config.controllerClass = Controller # do not change unless using a custom controller in order to override "_load_controller_model"
vehicle_config.controller_model_path = 'resources/controller_models/3b' # may be given as an absolute path, or as a relative path, in relation to the LiteRacer package directory
                                                                        # may use None for training mode (will not load a controller)
vehicle_config.controller_model_type = 'SAC' # stable_baselines model type. may use None for training mode (will not load a controller) or if using a custom model
vehicle_config.control_duration = 1 # (sec.) detemines the frequency of choosing new control actions
vehicle_config.delta_t = 0.25 # (sec.) time discretization resolution for continuous motion simulation.
               # reduce value to increase smoothness (vs. computation speed). use same value as control_duration to ignore
vehicle_config.run_timeout = 10000 # maximal amount of control steps per run

### sensor config
vehicle_config.sensing_freq = SensingFreq.CONTINUOUS #SensingFreq.ON_REQUEST # the frequency in which sensing is invoked
vehicle_config.sensorClass = Sensor # do not change unless overriding functionality
vehicle_config.sensor_angle_range = [-2*pi/5, 2*pi/5]
vehicle_config.sensor_max_range = 2
vehicle_config.sensor_dpi = 100 # base value is 100, increase to increase sensor "sensitivity" (and observation visulizer resolution)
vehicle_config.observation_image_size = [100, 50] # downsampled resolution of the observation image (input to controller).
                                   # original size is [2*sensor_max_range, sensor_max_range]*sensor_dpi
## set relative poistion of sensor (frame) in vehicle (frame, whose origin is located in rear-center)
vehicle_config.sensor_origin_offset_x = 0*vehicle_config.length # use 0 to overlap with vehicle frame, use "length" to place sensor in the front-center
vehicle_config.sensor_origin_offset_y = 0


"""Visualizer parameters."""
visualizer_config = SimpleNamespace()
visualizer_config.window_position_x = 0 # determines where the visualizer window opens on screen
visualizer_config.window_position_y = 0
visualizer_config.draw_observation_in_visualizer = DrawObservationInVisualizer.CURRENT
visualizer_config.open_separate_observation_view = True # when visualizer is opened
visualizer_config.visualizer_dpi = 100 # base value is 100 - increase value to increase visulizer resolution
visualizer_config.env_min_x_range = [0,1]
visualizer_config.env_min_y_range = [-2,2]