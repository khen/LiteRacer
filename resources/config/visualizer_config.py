"""Visualizer parameters."""

from utils.enums import Visualization, DrawObservationInVisualizer


visualization = Visualization.ON_AND_NO_BLOCK_ON_TERMINATION
window_position_x = 0 # determines where the visualizer window opens on screen
window_position_y = 0
draw_observation = DrawObservationInVisualizer.ON # only if visualization is ON
open_observation_view = True
visualizer_dpi = 100 # base value is 100 - increase value to increase visulizer resolution
min_x_range = [0,1]
min_y_range = [-2,2]