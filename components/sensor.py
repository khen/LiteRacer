"""Sensor."""

from math import pi, cos, sin
import shapely.geometry as sg
from shapely import plotting
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import gc


class Sensor():
    """Radar sensor."""

    _SF_sensing_wedge = None
    """Shape of sensing wedge (static to avoid recalculation)."""

    # CTOR
    def __init__(self, simulator, vehicle_config):

        self._simulator = simulator
        self._observation = None # a list of detected shapes in sensor frame [track, finish line]
        self._observation_listeners = []

        # init plotting environment for observations (required to create observation images)
        # must actively close fig window when sensor is deactivated
        sensor_frame_width = vehicle_config.sensor_max_range
        sensor_frame_height = 2*vehicle_config.sensor_max_range
        # self._sensor_fig = plt.figure(figsize=[fig_w,fig_h+0.25], dpi=100)
        self._sensor_fig = plt.figure(figsize=[sensor_frame_width,sensor_frame_height], dpi=vehicle_config.sensor_dpi)
        # self._sensor_ax = self._sensor_fig.add_axes([0, 0, 1, fig_h/(fig_h+0.25)])
        self._sensor_ax = self._sensor_fig.add_axes([0, 0, 1, sensor_frame_height/sensor_frame_height])
        
        fig_manager = plt.get_current_fig_manager()
        fig_manager.set_window_title("LiteRacer: Sensor View")

        self._sensor_max_range = vehicle_config.sensor_max_range
        self._angle_range = vehicle_config.sensor_angle_range
        if Sensor._SF_sensing_wedge is None:
            Sensor._SF_sensing_wedge = Sensor._calc_SF_sensing_wedge(self._sensor_max_range, self._angle_range)
        self._plot_observation()

        # cached variables for efficient observation plotting
        self._dynmic_elements_in_figure = []
        self._latest_observation_image = None

    def get_observation(self):
        """Get latest observation."""
        return self._observation

    def set_observation(self, new_observation):
        """Update the observation and invoke listener."""

        if self._observation != new_observation:
            # update observation
            self._observation = new_observation

            # clear previous cached observation image
            self._latest_observation_image = None

            # update the observation fig
            self._update_observation_plot()

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
            self._sensor_fig.canvas.manager.window.move(x,y)
        self._sensor_fig.show()

    def close_sensor_view(self):
        """Close observation visualizer."""

        plt.close(self._sensor_fig)

    def sense(self):
        """Return an observation (and set the appropriate variable) based on the world state from the simulator."""

        # ask for observation from the simulator, given the sensor parameters
        try:
            SF_observed_track_shape, SF_observed_finish_line_shape, x = self._simulator.calc_observed_shapes_in_SF(Sensor._SF_sensing_wedge,self._sensor_max_range)
        except Exception as e:
            # raise Exception("Observation failed. Probably sensor ovelaps with an obstacle.")
            self.set_observation(None)
            return None

        # update the observation
        self.set_observation([ SF_observed_track_shape, SF_observed_finish_line_shape ])

        return self._observation

    def get_observation_image(self, FROM_SCRATCH=False):
        """Return an image of the latest observation, or a given observation."""

        if self._latest_observation_image is not None:
            # return the chached image, if observation has no changed since last call 
            return self._latest_observation_image

        # plot observation from scratch on axis before saving the image
        if FROM_SCRATCH:
            self._plot_observation()

        # create image from fig
        image_buffer = BytesIO()        
        self._sensor_fig.savefig(image_buffer, format='png', bbox_inches='tight', pad_inches=0)
        image_buffer.seek(0)
        observation_image = Image.open(image_buffer)

        # save the image to file
        # import time
        # observation_image.save(f'./obs/{round(time.time() * 1000)}.png')

        # cache the observation image to avoid recalculating it on future function calls
        self._latest_observation_image = observation_image
        
        return observation_image

    def kill(self):
        """Release resources."""

        self._observation_listeners = []

        if self._sensor_fig is not None:
            self._sensor_ax.cla()
            self._sensor_fig.clf()
            plt.close(self._sensor_fig)
            del self._sensor_fig
            del self._sensor_ax
            gc.collect()

    def _plot_observation(self):
        """Plot from scratch latest observation on the sensor axis."""
        
        # reset axis
        self._sensor_ax.cla()
        # self._sensor_ax.title.set_text("Egocentric View")
        self._sensor_ax.set_axis_off()
        self._sensor_ax.set_xlim([0, self._sensor_max_range])
        self._sensor_ax.set_ylim([-self._sensor_max_range, self._sensor_max_range])

        # plot constant shapes
        SF_sensing_bounding_box = sg.Polygon([(0,-self._sensor_max_range),(0,self._sensor_max_range),(self._sensor_max_range,self._sensor_max_range),(self._sensor_max_range,-self._sensor_max_range)])
        self._sensor_ax.add_patch(plotting.patch_from_polygon(SF_sensing_bounding_box, fc='k', alpha=0.85))
        
        SF_out_of_range_shape = SF_sensing_bounding_box.difference(Sensor._SF_sensing_wedge)
        self._sensor_ax.add_patch(plotting.patch_from_polygon(SF_out_of_range_shape, fc='k'))

        # plot dynamic shapes
        self._plot_dynamic_elements()

        self._sensor_fig.canvas.draw()
        self._sensor_fig.canvas.flush_events()

    def _update_observation_plot(self):
        """Update the obsrvation plot based on the latest observation (removes old and adds new dyanmic objects)."""

        for element in self._dynmic_elements_in_figure:
            element.remove()
        self._dynmic_elements_in_figure = []

        self._plot_dynamic_elements()
        
        self._sensor_fig.canvas.draw()
        self._sensor_fig.canvas.flush_events()

    def _plot_dynamic_elements(self):
        """Add dyanmic objects on the obsrvation plot based on the given observation."""

        if self._observation is None:
            return

        SF_observed_track_shape = self._observation[0]
        SF_observed_finish_line_shape = self._observation[1]
        # observed track
        if not SF_observed_track_shape.is_empty and (SF_observed_track_shape.geom_type == 'Polygon' or SF_observed_track_shape.geom_type == 'MultiPolygon'):
            self._dynmic_elements_in_figure.append(self._sensor_ax.add_patch(plotting.patch_from_polygon(SF_observed_track_shape, linewidth=0, fc='w', alpha=0.6)))

        # observed finish line
        if not SF_observed_finish_line_shape.is_empty and (SF_observed_finish_line_shape.geom_type == 'Polygon' or SF_observed_finish_line_shape.geom_type == 'MultiPolygon'):
            self._dynmic_elements_in_figure.append(self._sensor_ax.add_patch(plotting.patch_from_polygon(SF_observed_finish_line_shape, linewidth=0, fc='w')))
            
    @staticmethod
    def _calc_SF_sensing_wedge(sensor_max_range, angle_range):
        """Return the shape of the sensing wedge in the sensor frame based on sensor parameters."""

        half_angle = (angle_range[1]-angle_range[0])/2

        bounding_triang_ray_length = (1/cos(half_angle))*sensor_max_range
        SF_sensing_bounding_triangle = sg.Polygon([(0,0), (bounding_triang_ray_length*cos(angle_range[0]),bounding_triang_ray_length*sin(angle_range[0])), \
                                                        (bounding_triang_ray_length*cos(angle_range[1]),bounding_triang_ray_length*sin(angle_range[1]))])
        return SF_sensing_bounding_triangle.intersection(sg.Point(0,0).buffer(sensor_max_range))
