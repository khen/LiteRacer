"""Visualizer parameters."""

from utils.enums import DrawObservationInVisualizer


window_position_x = 0 # determines where the visualizer window opens on screen
window_position_y = 0
draw_observation_in_visualizer = DrawObservationInVisualizer.CURRENT
open_separate_observation_view = True # when visualizer is opened
visualizer_dpi = 100 # base value is 100 - increase value to increase visulizer resolution
env_min_x_range = [0,1]
env_min_y_range = [-2,2]