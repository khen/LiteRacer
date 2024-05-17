"""Environment parameters."""

from math import sin, cos, pi
from scipy.special import ellipeinc

### obstacle parametes
# list of circular obstacles in format [c_x,c_y,r] # TODO: support obstacles of othre shapes
predefined_obstacles = [] # [[4, -1.8, 0.6], [2, 1.5 , 0.4]]
number_of_random_obstacles_at_init = 5 # number of obstacles to add to world on initialization
radius_of_random_obstacles = 0.06
start_zone_portion = 0.1 # portion of track from start line without obstacles

### track parameters: a padded area aroud a given y=f(x) curve # TODO: support any parametarized curve
X_track_range = [0, 3*pi]
track_padding = 0.6 # TODO: if road is very thick, boundary can twist and should be smoothed
track_percision = 0.02
finish_line_portion = 0.05 # portion of the track to be considered finish line area
## track curve definition (requires first and second derivative to calc padded area)
track_vert_scale = 0.8
track_func         = lambda x:  track_vert_scale*sin(x)
track_drv_func     = lambda x:  track_vert_scale*cos(x)
track_sec_drv_func = lambda x: -track_vert_scale*sin(x)
track_arc_length   = lambda x: ((1+track_vert_scale**2)**0.5) * ellipeinc(x, (1+track_vert_scale**2)**(-0.5))