"""Sensor."""

from math import pi, cos, sin
from numpy import sqrt
import shapely.geometry as sg
from shapely import plotting
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import gc
from ..utils.funcs import move_figure
from ..utils.enums import SensingFreq


class Sensor():
    """Radar sensor."""

    # CTOR
    def __init__(self, simulator, vehicle_config):

        self._simulator = simulator
        self._observation = None # a list of detected shapes in sensor frame [track, finish line]
        self._observation_listeners = []
        self.max_range = vehicle_config.sensor_max_range
        self.angle_range = vehicle_config.sensor_angle_range
        self.__sensing_freq = vehicle_config.sensing_freq # only to optimize performance

        # init plotting environment for observations (required to create observation images)
        # must actively close fig window when sensor is deactivated
        sensor_frame_width = vehicle_config.sensor_max_range
        sensor_frame_height = 2*vehicle_config.sensor_max_range
        # self._sensor_fig = plt.figure(figsize=[fig_w,fig_h+0.25], dpi=100)
        self._sensor_fig = plt.figure(figsize=[sensor_frame_width,sensor_frame_height], dpi=vehicle_config.sensor_dpi)
        # self._sensor_ax = self._sensor_fig.add_axes([0, 0, 1, fig_h/(fig_h+0.25)])
        self.__sensor_ax = self._sensor_fig.add_axes([0, 0, 1, sensor_frame_height/sensor_frame_height])
        
        fig_manager = plt.get_current_fig_manager()
        fig_manager.set_window_title("LiteRacer: Sensor View")

        self._SF_sensing_wedge = self.__calc_SF_sensing_wedge()
        self.__plot_observation()

        # cached variables for efficient observation plotting
        self.__dynmic_elements_in_figure = []
        self.__latest_observation_image = None

    @property
    def observation(self):
        """Get latest observation."""
        return self._observation

    @observation.setter
    def observation(self, new_observation):
        """Update the observation and invoke listener."""

        if self._observation != new_observation:
            # update observation
            self._observation = new_observation

            # clear previous cached observation image
            self.__latest_observation_image = None

            # update the observation fig
            self.__update_observation_plot()

            # invoke listening functions
            for listener in self._observation_listeners:
                listener()

    def add_listener_observation(self, func, override_priority=False):
        """Add listener to observation changes."""

        if override_priority:
            self._observation_listeners.insert(0, func)
        else:
            self._observation_listeners.append(func)

    def open_sensor_view(self, x=None, y=None):
        """Open observation visualizer."""

        if x is not None and y is not None:
            if x!=0 or y!=0:
                move_figure(self._sensor_fig, x, y)
        self._sensor_fig.show()

    def close_sensor_view(self):
        """Close observation visualizer."""

        plt.close(self._sensor_fig)

    def sense(self):
        """Return an observation (and set the appropriate variable) based on the simulation state."""

        # ask for observation from the simulator, given the sensor parameters
        try:
            SF_observed_track_shape, SF_observed_finish_line_shape, x = self._simulator.calc_observed_shapes_in_SF(self._SF_sensing_wedge,self.max_range)
        except Exception as e:
            # raise Exception("Observation failed. Probably sensor ovelaps with an obstacle.")
            self.observation = None
            return None

        # update the observation
        self.observation = [ SF_observed_track_shape, SF_observed_finish_line_shape ]

        return self._observation

    def get_observation_image(self, FROM_SCRATCH=False):
        """Return an image of the latest observation, or a given observation."""

        if self.__sensing_freq == SensingFreq.ON_REQUEST:
            # sensing is not called automatically, so must call manually
            self.sense()

        if self.__latest_observation_image is not None:
            # return the chached image, if observation has no changed since last call 
            return self.__latest_observation_image

        # plot observation from scratch on axis before saving the image
        if FROM_SCRATCH:
            self.__plot_observation()

        # create image from fig
        # image_buffer = BytesIO()        
        # self._sensor_fig.savefig(image_buffer, format='png', bbox_inches='tight', pad_inches=0)
        # image_buffer.seek(0)
        # observation_image = Image.open(image_buffer)

        import numpy as np

        image_flat = np.frombuffer(self._sensor_fig.canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)

        im_size = self._sensor_fig.canvas.get_width_height()
        scale_correction = sqrt(len(image_flat)/(im_size[0]*im_size[1]*3))
        corrected_im_size = [int(scale_correction*im_size[0]), int(scale_correction*im_size[1])]

        # NOTE: reversed converts (W, H) from get_width_height to (H, W)
        observation_image_array = image_flat.reshape(*reversed(corrected_im_size), 3)  # (H, W, 3)

        from PIL import Image
        observation_image = Image.fromarray(observation_image_array) # RGB image

        # save the image to file
        # import time
        # observation_image.save(f'./obs/{round(time.time() * 1000)}.png')

        # cache the observation image to avoid recalculating it on future function calls
        self.__latest_observation_image = observation_image
        
        return observation_image

    def kill(self):
        """Release resources."""

        self._observation_listeners = []
        self.close_sensor_view()

    def __plot_observation(self):
        """Plot from scratch latest observation on the sensor axis."""
        
        # reset axis
        self.__sensor_ax.cla()
        # self._sensor_ax.title.set_text("Egocentric View")
        self.__sensor_ax.set_axis_off()
        self.__sensor_ax.set_xlim([0, self.max_range])
        self.__sensor_ax.set_ylim([-self.max_range, self.max_range])

        # plot constant shapes
        SF_sensing_bounding_box = sg.Polygon([(0,-self.max_range),(0,self.max_range),(self.max_range,self.max_range),(self.max_range,-self.max_range)])
        self.__sensor_ax.add_patch(plotting.patch_from_polygon(SF_sensing_bounding_box, fc='k', alpha=0.85))
        
        SF_out_of_range_shape = SF_sensing_bounding_box.difference(self._SF_sensing_wedge)
        self.__sensor_ax.add_patch(plotting.patch_from_polygon(SF_out_of_range_shape, fc='k'))

        # plot dynamic shapes
        self.__plot_dynamic_elements()

        self._sensor_fig.canvas.draw()
        self._sensor_fig.canvas.flush_events()

    def __update_observation_plot(self):
        """Update the obsrvation plot based on the latest observation (removes old and adds new dyanmic objects)."""

        for element in self.__dynmic_elements_in_figure:
            element.remove()
        self.__dynmic_elements_in_figure = []

        self.__plot_dynamic_elements()
        
        self._sensor_fig.canvas.draw()
        self._sensor_fig.canvas.flush_events()

    def __plot_dynamic_elements(self):
        """Add dyanmic objects on the obsrvation plot based on the given observation."""

        if self._observation is None:
            return

        SF_observed_track_shape = self._observation[0]
        SF_observed_finish_line_shape = self._observation[1]
        # observed track
        if not SF_observed_track_shape.is_empty and (SF_observed_track_shape.geom_type == 'Polygon' or SF_observed_track_shape.geom_type == 'MultiPolygon'):
            self.__dynmic_elements_in_figure.append(self.__sensor_ax.add_patch(plotting.patch_from_polygon(SF_observed_track_shape, linewidth=0, fc='w', alpha=0.6)))

        # observed finish line
        if not SF_observed_finish_line_shape.is_empty and (SF_observed_finish_line_shape.geom_type == 'Polygon' or SF_observed_finish_line_shape.geom_type == 'MultiPolygon'):
            self.__dynmic_elements_in_figure.append(self.__sensor_ax.add_patch(plotting.patch_from_polygon(SF_observed_finish_line_shape, linewidth=0, fc='w')))
            
    
    def __calc_SF_sensing_wedge(self):
        """Return the shape of the sensing wedge in the sensor frame based on sensor parameters."""

        angle_range = self.angle_range
        half_angle = (angle_range[1]-angle_range[0])/2

        bounding_triang_ray_length = (1/cos(half_angle))*self.max_range
        SF_sensing_bounding_triangle = sg.Polygon([(0,0), (bounding_triang_ray_length*cos(angle_range[0]),bounding_triang_ray_length*sin(angle_range[0])), \
                                                        (bounding_triang_ray_length*cos(angle_range[1]),bounding_triang_ray_length*sin(angle_range[1]))])
        return SF_sensing_bounding_triangle.intersection(sg.Point(0,0).buffer(self.max_range))
