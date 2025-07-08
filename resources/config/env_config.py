"""Environment parameters."""

from math import sin, cos, pi
from scipy.special import ellipeinc


### track parameters: a padded area aroud a given y=f(x) curve # TODO: support any parametarized curve
X_track_range = [0, 5*pi]
track_padding = 0.8 # TODO: if road is very thick, boundary can twist and should be smoothed
track_percision = 0.02
start_zone_portion = 0.1 # portion of track from start line without (random) obstacles
finish_line_portion = 0.1 # portion of the track to be designated finish line area--MAKE SURE to provide an area big enough to contain the vehicle
## track curve definition (requires first and second derivative to calc padded area)
track_vert_scale = 0.8
track_func         = lambda x:  track_vert_scale*sin(x)
track_drv_func     = lambda x:  track_vert_scale*cos(x)
track_sec_drv_func = lambda x: -track_vert_scale*sin(x)
track_arc_length   = lambda x: ((1+track_vert_scale**2)**0.5) * ellipeinc(x, (1+track_vert_scale**2)**(-0.5))


### obstacle parametes
# list of circular obstacles in format [c_x,c_y,r] # TODO: support obstacles of other shapes
    # c_x/c_y are to be specified a fraction in the range [0,1] 
    # c_x is the position along the the trajectory curve (whose x range is X_track_range), excluding the state_zone/finish_line portions.
    # c_y is the position between the long lower boundary and the upper boundary (on the axis perpendicular to the track curve on given x)
predefined_obstacles = [[0.1, 0.3, 0.43], [0.8, 0.8 , 0.25]]
number_of_random_obstacles_at_init = 3 # number of obstacles to add to env on initialization
radius_of_random_obstacles = 0.1