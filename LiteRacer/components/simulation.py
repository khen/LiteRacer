"""World simulator."""

import gc
from math import sqrt, acos, atan2, sin, cos, pi, inf
from numpy import arange, infty, vectorize
from matplotlib import patches
import matplotlib.pyplot as plt
from shapely import affinity, plotting
import shapely.geometry as sg
import random
from ..utils.funcs import min_distance_from_curve, move_figure
from ..utils.enums import DrawObservationInVisualizer, VehicleStatus
from .vehicle import Vehicle


class Simulation():
    """World simulator coordinating the interaction of a vehicle in an environment."""

    # static track shapes/info, to avoid caluclating from scratch for each copy
    _track_shape = None
    """Shape of track."""

    def __init__(self, override_initial_state=None, env_config=None, vehicle_config=None, visualizer_config=None, init_obstacles=True):
        """Initiate a Simulation instance with a single car in a specified environment, according to given configuration."""

        # load config
        if env_config is None:
            from ..resources.config import env_config
        self.env_config = env_config

        if vehicle_config is None:
            from ..resources.config import vehicle_config
        self.vehicle_config = vehicle_config

        if visualizer_config is None:
            from ..resources.config import visualizer_config
        self.visualizer_config = visualizer_config

        # calc static shapes on first intitalization
        if Simulation._track_shape is None:
            self._calc_track_shapes(self.env_config)

        # init track related config
        self._env_bounding_box =   [[min([self.visualizer_config.env_min_x_range[0],min(Simulation._X_track_curve)-self.env_config.track_padding]),\
                                     min([self.visualizer_config.env_min_y_range[0],min(Simulation._Y_track_curve)-self.env_config.track_padding])],\
                                    [max([self.visualizer_config.env_min_x_range[1],max(Simulation._X_track_curve)+self.env_config.track_padding]),\
                                     max([self.visualizer_config.env_min_y_range[1],max(Simulation._Y_track_curve)+self.env_config.track_padding])]]
                                     # bottom-left, top-right

        # cached dynamic shapes that are updated with vehicle movement
        self._dynmic_elements_in_figure = []
        self._observed_sensing_wedge = sg.Point() # these are only practically updated when sensor asks for observed shapes
        self._observed_track_shape = sg.Point()
        self._observed_finish_line_shape = sg.Point()
        self._observed_region_shape = sg.Point()
        self._latest_sensing_wedge = None
        self._obst_track_shape = sg.Polygon([[self._env_bounding_box[0][0],self._env_bounding_box[0][1]], \
                                             [self._env_bounding_box[1][0],self._env_bounding_box[0][1]], \
                                             [self._env_bounding_box[1][0],self._env_bounding_box[1][1]], \
                                             [self._env_bounding_box[0][0],self._env_bounding_box[1][1]]]) \
                                             .difference(Simulation._track_shape)
        #self._unobst_track_shape = Simulation._track_shape
        # THIS IS A TRICK SO I CAN USE THIS SHAPE TO CALC DISTANCE TO COLLISION, OTHERWISE TRACK EDGE IS COUNTED
        self._unobst_track_shape = Simulation._track_shape.union(sg.Point(0,0).buffer(self.env_config.track_padding))

        # init plotting environment
        self._simulation_fig = None
        self._fig_w = self._env_bounding_box[1][0]-self._env_bounding_box[0][0]
        self._fig_h = self._env_bounding_box[1][1]-self._env_bounding_box[0][1]
        self._is_simulation_fig_open = False

        # init vehicle
        self._vehicle_state_history = []
        self._vehicle_shape_history = []
        self._observation_wedges_history = []
        if override_initial_state is not None:
            self._initial_vehicle_state = override_initial_state
        else:
            self._initial_vehicle_state = self.vehicle_config.initial_state
        vehicle = Vehicle(self, self.vehicle_config, self._initial_vehicle_state)
        self.reset_vehicle(vehicle, call_sense_after_reset=False) # will call sense manually after obstacle initialization

        # init obstacles (after vehicle, to be able to veify that random obstacles do not overlap with its inital position)
        self.obstacles_in_WF = []
        self._obstacle_circles = []
        if init_obstacles:
            self.add_obstacles_relative_to_track(self.env_config.predefined_obstacles)
            if self.env_config.number_of_random_obstacles_at_init > 0:
                self.add_random_obstacles(self.env_config.number_of_random_obstacles_at_init)

        # sense to  observe new obstacles
        self.vehicle.sensor.sense() # only need to call "sense" the first time. It is later automatically triggered by the state change
        self._observation_wedges_history.append(self._latest_sensing_wedge)

    def rollback(self, to_timestamp, call_sense_after_rollback=True):
        """Rollback the vehicle to a given point in the past."""
        
        to_timestamp = max([0,to_timestamp])

        if to_timestamp < len(self._vehicle_state_history):
                vehicle = Vehicle(self, self.vehicle_config, self._vehicle_state_history[to_timestamp])

        self._vehicle_state_history = self._vehicle_state_history[0:to_timestamp]
        self._vehicle_shape_history = self._vehicle_shape_history[0:to_timestamp]
        self._observation_wedges_history =  self._observation_wedges_history[0:to_timestamp]
        self._latest_sensing_wedge = None # to make sure we do not add twice this shape to history
        self._observed_region_shape = sg.Point() # TODO: rollback observed region shape as well (currently resets it)

        self.vehicle.kill()
        del self.vehicle
        self.vehicle = vehicle
            
        # update the simulation according to the new vehicle's state 
        self._vehicle_state_changed_listener_initial()
        if call_sense_after_rollback:
            self.vehicle.sensor.sense()
        self._vehicle_state_changed_listener_final()

        # listen to future state changes
        self.vehicle.add_listener_state(self._vehicle_state_changed_listener_initial, override_priority=True)
        self.vehicle.add_listener_state(self._vehicle_state_changed_listener_final)
     
    def reset_vehicle(self, vehicle=None, call_sense_after_reset=True):
        """Reset the vehicle to a given state (or the initial state, if none is given). Clears vehicle history."""

        if vehicle is None: # reset to initial vehicle state
            vehicle = Vehicle(self, self.vehicle_config, self._initial_vehicle_state)

        self.vehicle = vehicle

        self._vehicle_state_history = []
        self._vehicle_shape_history = []
        self._observation_wedges_history = []
        self._latest_sensing_wedge = None # to make sure we do not add twice this shape to history
        self._observed_region_shape = sg.Point()

        # update the simulation according to the new vehicle's state 
        self._vehicle_state_changed_listener_initial()
        if call_sense_after_reset:
            self.vehicle.sensor.sense() # only need to call "sense" the first time. It is later automatically triggered by the state change
        self._vehicle_state_changed_listener_final()

        # listen to future state changes
        self.vehicle.add_listener_state(self._vehicle_state_changed_listener_initial, override_priority=True)
        self.vehicle.add_listener_state(self._vehicle_state_changed_listener_final)

    def copy(self):
        """Create a copy of current Simulation."""
        # create basic copy without obstacles
        sim_copy = type(self)(self.vehicle.get_state(), self.env_config, self.vehicle_config, self.visualizer_config, init_obstacles=False)
        # copy obstacles
        sim_copy.add_obstacles_in_WF(self.obstacles_in_WF)

        # copy history-built information
        sim_copy._initial_vehicle_state = self._initial_vehicle_state
        sim_copy._vehicle_state_history = self._vehicle_state_history
        sim_copy._vehicle_shape_history = self._vehicle_shape_history
        sim_copy._observation_wedges_history = self._observation_wedges_history
        sim_copy._observed_region_shape = sg.Polygon(self._observed_region_shape)
        sim_copy.vehicle.sensor.sense() # preactically copies the latest observation in the sensor and the observation-related shapes in the simulation copy
        self._update_vehicle_status() # recalculate the status given the added obstacles

        return sim_copy

    def kill(self):
        """Release resources."""
        
        self.vehicle.kill()

        if self._simulation_fig is not None:
            self.close_visualizer()
   
    def open_visualizer(self, x=None, y=None, open_separate_observation_view=None):
        """Open simulation visualizer."""

        if x is None:
            x = self.visualizer_config.window_position_x
        if y is None:
            y = self.visualizer_config.window_position_y
        if open_separate_observation_view is None:
            open_separate_observation_view = self.visualizer_config.open_separate_observation_view
        
        # init plotting environment
        self._simulation_fig = plt.figure(figsize=[self._fig_w,self._fig_h], dpi=self.visualizer_config.visualizer_dpi)
        self._simulation_ax = self._simulation_fig.add_axes([0.0, 0, 1.0, 1])

        fig_manager = plt.get_current_fig_manager()
        fig_manager.set_window_title("LiteRacer: World View")

        if x!=0 or y!=0:
            move_figure(self._simulation_fig, x, y)

        self._plot_simulation()
        self._simulation_fig.show()

        # track if figure is open or close (to know if should update plot)
        self._is_simulation_fig_open = True
        def on_simulation_fig_close(event):
            # event.canvas.figure.axes[0].is_open = False
            self._is_simulation_fig_open = False
        self._simulation_fig.canvas.mpl_connect('close_event', on_simulation_fig_close)

        if open_separate_observation_view == True:
            simulation_fig_size = self._simulation_fig.get_size_inches()*self._simulation_fig.dpi#/2
            self.vehicle.sensor.open_sensor_view(x+int(simulation_fig_size[0]),y)

    def close_visualizer(self):
        """Closes visualizer."""

        if self._simulation_fig is not None:
            plt.close(self._simulation_fig)

    def add_obstacles_relative_to_track(self, new_obstacles, verify_intersection_with_vehicle=True):
        """Add given obstacles to the simulation. Rreturns a list of the obstacles successfully added with positions in WF."""
        
        obstacles_added = []
        for i in range(len(new_obstacles)):
            new_obs = self._add_obstacle_relative_to_track(new_obstacles[i], verify_intersection_with_vehicle)
            if new_obs is not None:
                obstacles_added.append(new_obs)

        return obstacles_added

    def add_obstacles_in_WF(self, new_obstacles, verify_intersection_with_vehicle=True):
        """Add given obstacles to the simulation. Rreturns a list of the obstacles successfully added with positions in WF."""
        
        obstacles_added = []
        for i in range(len(new_obstacles)):
            new_obs = self._add_obstacle_in_WF(new_obstacles[i], verify_intersection_with_vehicle)
            if new_obs is not None:
                obstacles_added.append(new_obs)

        return obstacles_added
        
    def _add_obstacle_relative_to_track(self, new_obstacle, verify_intersection_with_vehicle):
        """Add an obstacle in a requested position in relation to the track. Return the obstacle position in WF (or None, if failed)."""

        x_along_track, y_along_track, radius = new_obstacle[0], new_obstacle[1],new_obstacle[2]

        # find requested position on the track as [x,y]
        x_length = len(self._track_boundary)/2
        min_x = round(x_length*self.env_config.start_zone_portion)
        max_x = round(x_length*(1-self.env_config.finish_line_portion)-radius/self.env_config.track_percision)

        selected_x = round(min_x + x_along_track*(max_x-min_x))
        
        track_boundary_point1 = self._track_boundary[selected_x]
        track_boundary_point2 = self._track_boundary[-1-selected_x]

        new_obstacle_in_xy = [(y_along_track*track_boundary_point1[0]+(1-y_along_track)*track_boundary_point2[0]),\
                            (y_along_track*track_boundary_point1[1]+(1-y_along_track)*track_boundary_point2[1]), radius]
        
        # add obstacle in [x,y] position
        succuss = self._add_obstacle_in_WF(new_obstacle_in_xy, verify_intersection_with_vehicle)
        if succuss:
            return new_obstacle_in_xy
        else:
            return None
         
    def _add_obstacle_in_WF(self, new_obstacle, verify_intersection_with_vehicle):
        """Add a given circular obstacle to the simulation. Return success flag."""
        
        obst_center_x, obst_center_y, obst_r = new_obstacle[0], new_obstacle[1],new_obstacle[2]
        obstacle_circle = sg.Point(obst_center_x,obst_center_y).buffer(obst_r)
        # if obstacle is legal, i.e., does not intersect with vehicle, add it, otherwise, abort
        if verify_intersection_with_vehicle == False or obstacle_circle.intersection(self._vehicle_shape).is_empty:
            self._obstacle_circles = self._obstacle_circles + [ obstacle_circle ]
            self._obst_track_shape = self._obst_track_shape.union(obstacle_circle)
            self._unobst_track_shape = self._unobst_track_shape.difference(obstacle_circle)
            self.obstacles_in_WF = self.obstacles_in_WF + [ new_obstacle ]
            success = True
        else:
            success = False
        
        return success
    
    def add_random_obstacles(self, num_of_obs_to_add, override_radius=None, verify_intersection_with_vehicle=True):
        """Add multiple new random obstacles. Return a list of the added obstacles in WF."""

        if override_radius is None:
            override_radius = self.env_config.radius_of_random_obstacles
        obstacles_added = []
        for i in range(num_of_obs_to_add): 
            new_obs = self._add_obstacle_relative_to_track([random.random(), random.random(), override_radius], verify_intersection_with_vehicle)
            if new_obs is not None:
                obstacles_added.append(new_obs)

        return obstacles_added

    def _add_random_obstacle(self, radius, verify_intersection_with_vehicle):
        """Add an obstacle in a random location on the track. Return the added obstacle in WF."""

        while True: # sample positions and try to add a random obstacle, until a legal one is found
            # sample a random position on the track
            x_length = len(self._track_boundary)/2
            random_x = random.randint(round(x_length*self.env_config.start_zone_portion),\
                                      round(x_length*(1-self.env_config.finish_line_portion)-radius/self.env_config.track_percision))

            track_boundary_point1 = self._track_boundary[random_x]
            track_boundary_point2 = self._track_boundary[-1-random_x]

            random_y = random.random()

            random_position_in_xy = [(random_y*track_boundary_point1[0]+(1-random_y)*track_boundary_point2[0]),\
                               (random_y*track_boundary_point1[1]+(1-random_y)*track_boundary_point2[1]), radius]
            
            # TODO: add support to "PRESERVE_HISTORY" option
            # if PRESERVE_HISTORY and self._observed_region_shape.distance(sg.Point(random_position)) <= radius:
            #     # if the added obstacle is going to overlap with/compromise the observed history
            #     continue
            
            # TODO: support adding obstacle in a constrained region
            # calculate region to sample in... ()
            # sample withing that region
            # radius_correction = radius/self.env_config.track_percision
            # region_start_x = round(x_length*region_start_perc+radius_correction)
            # region_end_x = round(x_length*region_end_perc-radius_correction)
            # random_x = random.randint(region_start_x, region_end_x)
            # random_points_in_polygon()

            succuss = self._add_obstacle_in_xy_postition(random_position_in_xy, verify_intersection_with_vehicle)
            if succuss: # obstacle was legal
                return random_position_in_xy
            else:
                continue

    def remove_obstacles(self, obs_indices):
        """Remove obstacles at given indices from obstacle list."""

        obstacles_removed = []
        for obs_index in obs_indices:
            obstacles_removed.append(self._remove_obstacle(obs_index)) 
        return obstacles_removed

    def _remove_obstacle(self, obs_index):
        """Remove an obstacle at given index from obstacle list."""

        obstacle_removed = self.obstacles_in_WF[obs_index]
        obstacle_circle = self._obstacle_circles[obs_index]

        # TODO: DOES NOT WORK WELL WHEN OBSTACLE IS PARTIALLY OUT OF TRACK SHAPE
        self._obst_track_shape = self._obst_track_shape.difference(obstacle_circle)
        self._unobst_track_shape = self._unobst_track_shape.union(obstacle_circle)

        del self.obstacles_in_WF[obs_index]
        del self._obstacle_circles[obs_index]
        
        return obstacle_removed

    def remove_random_obstacles(self, num_of_obs_to_remove):
        """Remove multiple random obstacles from obstacle list."""

        obstacles_removed = []
        for i in range(num_of_obs_to_remove):
            # TODO: add "PRESERVE_HISTORY" option, to only remove obstacles which were not observed
            # (should keep a list to indicate for every circle if it was observed)

            # choose a random obstacle and remove it
            index = random.randint(0,len(self.obstacles_in_WF)-1)
            obstacles_removed.append(self._remove_obstacle(index)) 
        return obstacles_removed

    def get_vehicle_progress_along_track(self):
        """Calculate and return the vehicle's progress (precentage) along the track."""

        # default - calculated based on the vehicle frame of origin (back, center)
        
        state = self.vehicle.get_state()
        x = state[0]
        y = state[1]

        min_idxs, distances = min_distance_from_curve(Simulation._X_track_curve, Simulation._Y_track_curve, x, y)
        closest_x = Simulation._X_track_curve[min_idxs[-1]] # when multiple points, take the one furthest down the track
        return self.env_config.track_arc_length(closest_x)/Simulation._track_length


    def get_distance_to_collision_throughout_history(self, only_in_sensing_range=True): 
        """Return the minimal distance of vehicle throughout its trajectory from the obstacles/shoulders."""

        min_d = inf
        for i in range(len(self._vehicle_shape_history)):            
            d = self.get_distance_to_collision(i, only_in_sensing_range)
            if d < min_d:
                min_d = d
        return min_d

    def get_distance_to_collision(self, timestamp=-1, only_in_sensing_range=True):
        """Return the distance of vehicle (at given timestamp) from the obstacles/shoulders."""
        
        safety_boundary = self._unobst_track_shape.boundary
        vehicle_shape = self._vehicle_shape_history[timestamp]

        if only_in_sensing_range:
            observation_shape = self._observation_wedges_history[timestamp]
            observed_boundary = observation_shape.intersection(safety_boundary)
            if not observed_boundary.is_empty:
                d = vehicle_shape.distance(observed_boundary)
            else:
                d = 2*self.env_config.track_padding # upper bound
        else:
            d = vehicle_shape.distance(safety_boundary)

        return d

    def calc_observed_shapes_in_SF(self, SF_sensing_wedge, sensor_range):
        """Calculate and return the shapes representing the observation given the sensor information and vehicle location."""
        
        # calc observation in world frame
        self._observed_sensing_wedge, self._observed_track_shape, self._observed_finish_line_shape = self._calc_observed_shapes_in_WF(SF_sensing_wedge, sensor_range)

        # tranaform observation to sensor frame
        SF_observed_track_shape = self.vehicle.VF_to_SF(self._WF_to_VF(self._observed_track_shape))
        SF_observed_finish_line_shape = self.vehicle.VF_to_SF(self._WF_to_VF(self._observed_finish_line_shape))
        SF_observed_sensing_wedge = self.vehicle.VF_to_SF(self._WF_to_VF(self._observed_sensing_wedge))

        # updates the history -- must practically simplify (approximate) the polygon for performance issues...
        simplify_tolerance = .05
        self._observed_region_shape = self._observed_region_shape.union(self._observed_sensing_wedge).simplify(tolerance=simplify_tolerance, preserve_topology=True).buffer(0)

        return SF_observed_track_shape, SF_observed_finish_line_shape, SF_observed_sensing_wedge

    def _calc_observed_shapes_in_WF(self, SF_sensing_wedge, sensor_range):
        """Calculate the shapes representing the current observation, given the sensor information and vehicle location."""

        sensing_wedge = self._VF_to_WF(self.vehicle.SF_to_VF(SF_sensing_wedge))
        sensor_point = self._VF_to_WF(self.vehicle.VF_sensor_origin)
        
        self._latest_sensing_wedge = sensing_wedge # to maintain observation history

        # for every obstacle calc hidden areas behind it
        observed_sensing_wedge = sensing_wedge
        sensor_x, sensor_y = sensor_point.x, sensor_point.y
        for i in range(len(self._obstacle_circles)):
            observed_sensing_wedge = observed_sensing_wedge.difference(self._obstacle_circles[i])
            
            obst_center_x, obst_center_y, obst_r = self.obstacles_in_WF[i][0], self.obstacles_in_WF[i][1], self.obstacles_in_WF[i][2]
            
            # check if obstacle is close enough to be considered
            dist_SC = sqrt((sensor_x - obst_center_x)**2 + (sensor_y - obst_center_y)**2)
            if dist_SC <= sensor_range:
                # find tangent points on obstacle circle, in relation to current pose
                angle_between_SC_radius = acos(obst_r / dist_SC)
                S_angle_from_center = atan2(sensor_y - obst_center_y, sensor_x - obst_center_x)  # direction angle of sensor from center
                tp1_angle_from_C = S_angle_from_center + angle_between_SC_radius  # direction angle of tp1 from center
                tp2_angle_from_C = S_angle_from_center - angle_between_SC_radius  # direction angle of tp2 from center

                tp1_x = obst_center_x + obst_r * cos(tp1_angle_from_C)
                tp1_y = obst_center_y + obst_r * sin(tp1_angle_from_C)
                tp2_x = obst_center_x + obst_r * cos(tp2_angle_from_C)
                tp2_y = obst_center_y + obst_r * sin(tp2_angle_from_C)

                # an area behind the obstacle is hidden only if one of the tangent point is in the sensing wedge
                if sensing_wedge.contains(sg.Point(tp1_x,tp1_y)) or sensing_wedge.contains(sg.Point(tp2_x,tp2_y)):
                    # find bounding trapezoid of shape hidden behind obstacle
                    tp1_angle_from_S = atan2(tp1_y - sensor_y, tp1_x - sensor_x)  # direction angle of point tp1 from sensor
                    tp2_angle_from_S = atan2(tp2_y - sensor_y, tp2_x - sensor_x)  # direction angle of point tp2 from sensor
                    # make sure the direction of the angle is towards the obstacle
                    angle_between = abs(tp2_angle_from_S-tp1_angle_from_S)
                    if angle_between > pi:
                        angle_between = 2*pi-angle_between

                    bounding_triang_ray_length = (1/cos(angle_between/2))*sensor_range
                    bp1_x = sensor_x + bounding_triang_ray_length * cos(tp1_angle_from_S)
                    bp1_y = sensor_y + bounding_triang_ray_length * sin(tp1_angle_from_S)
                    bp2_x = sensor_x + bounding_triang_ray_length * cos(tp2_angle_from_S)
                    bp2_y = sensor_y + bounding_triang_ray_length * sin(tp2_angle_from_S)

                    hidden_shape_bounding_trapezoid = sg.Polygon([(tp2_x,tp2_y), (tp1_x,tp1_y), (bp1_x,bp1_y), (bp2_x,bp2_y)])
                    #global hidden_shape; hidden_shape = sensing_wedge.intersection(hidden_shape_bounding_trap)
                    observed_sensing_wedge = observed_sensing_wedge.difference(hidden_shape_bounding_trapezoid)

        observed_track_shape = self._unobst_track_shape.intersection(observed_sensing_wedge)
        observed_track_shape = observed_track_shape.difference(self._vehicle_shape)
        observed_finish_line_shape = self._finish_line_shape.intersection(observed_sensing_wedge)
        observed_finish_line_shape = observed_finish_line_shape.difference(self._vehicle_shape)

        return observed_sensing_wedge, observed_track_shape, observed_finish_line_shape

    def _plot_simulation(self):
        """Plots simulation from scratch."""
        
        self._simulation_ax.cla()
        # self._simulation_ax.title.set_text("World View")
        self._simulation_ax.set_aspect('equal')
        self._simulation_ax.set_xlim([self._env_bounding_box[0][0], self._env_bounding_box[1][0]])
        self._simulation_ax.set_ylim([self._env_bounding_box[0][1], self._env_bounding_box[1][1]])
        # remove annotaion of axes
        plt.tick_params(
            axis='both',       # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False) # labels along the bottom edge are off
        #self._simulation_fig.tight_layout()

        # track
        self._simulation_ax.plot(Simulation._X_track_curve, Simulation._Y_track_curve, color='k', linestyle='dotted', alpha=0.05)
        self._simulation_ax.add_patch(plotting.patch_from_polygon(Simulation._track_shape, linewidth=1, ec='k',fc='k', alpha=0.3))
        self._simulation_ax.add_patch(plotting.patch_from_polygon(Simulation._finish_line_shape, fc='k', alpha=0.5))

        # obstacles
        for i in range(len(self._obstacle_circles)):
            self._simulation_ax.add_patch(plotting.patch_from_polygon(self._obstacle_circles[i], linewidth=1, ec='k', fc='darkorange'))

        # vehicle history
        for vehicle_state in self._vehicle_state_history:
            last_vehicle_position = vehicle_state[0:2]
            self._simulation_ax.add_patch(patches.Circle((last_vehicle_position[0], last_vehicle_position[1]), radius=0.05, linewidth=0, fc='lightgreen', alpha=0.5))
        
        self._update_dynamic_elements_in_plot()

        self._simulation_fig.canvas.draw()
        self._simulation_fig.canvas.flush_events()

    def _update_plot(self):
        """Update the simulation plot based on the latest vehicle movement (remove old and add new dyanmic objects)."""

        # vehicle history
        last_vehicle_position = self._vehicle_state_history[-1][0:2]
        self._simulation_ax.add_patch(patches.Circle((last_vehicle_position[0], last_vehicle_position[1]), radius=0.05, linewidth=0, fc='lightgreen', alpha=0.5))
                
        for element in self._dynmic_elements_in_figure:
            element.remove()
        self._dynmic_elements_in_figure = []

        self._update_dynamic_elements_in_plot()

        self._simulation_fig.canvas.draw()
        self._simulation_fig.canvas.flush_events()

    def _update_dynamic_elements_in_plot(self):
        """Add dyanmic objects on the simulation plot based on the latest vehicle movement ."""

        # vehicle
        self._dynmic_elements_in_figure.append(self._simulation_ax.add_patch(plotting.patch_from_polygon(self._vehicle_shape, fc='lightgreen', ec='k')))

        # steering angle indicator
        arrow_point = self._VF_to_WF(sg.Point(self.vehicle.length-self.vehicle.width,0))
        absolute_steering_angle = self.vehicle._state[2]+self.vehicle._state[3]
        self._dynmic_elements_in_figure.append(self._simulation_ax.arrow(arrow_point.x,arrow_point.y, 0.01*cos(absolute_steering_angle), 0.01*sin(absolute_steering_angle),\
                                                head_width=self.vehicle.width/2, head_length=self.vehicle.width, linewidth=0, color='darkgreen'))

        # vehicle frame origin
        self._dynmic_elements_in_figure.append(self._simulation_ax.add_patch(patches.Circle((self.vehicle._state[0], self.vehicle._state[1]), radius=0.05, linewidth=0, fc='darkgreen')))

        if self.visualizer_config.draw_observation_in_visualizer == DrawObservationInVisualizer.CURRENT_AND_HISTORY:

            # sensing history
            if not self._observed_region_shape.is_empty:
                self._dynmic_elements_in_figure.append(self._simulation_ax.add_patch(plotting.patch_from_polygon(self._observed_region_shape, linewidth=0, color='r', alpha=0.05)))
                
        if self.visualizer_config.draw_observation_in_visualizer == DrawObservationInVisualizer.CURRENT_AND_HISTORY or self.visualizer_config.draw_observation_in_visualizer == DrawObservationInVisualizer.CURRENT:
            # observed track    
            if not self._observed_track_shape.is_empty and (self._observed_track_shape.geom_type == 'Polygon' or self._observed_track_shape.geom_type == 'MultiPolygon'):
                self._dynmic_elements_in_figure.append(self._simulation_ax.add_patch(plotting.patch_from_polygon(self._observed_track_shape, linewidth=0, fc='r', ec='k', alpha=0.2)))

            # observed finish line
            if not self._observed_finish_line_shape.is_empty and (self._observed_finish_line_shape.geom_type == 'Polygon' or self._observed_finish_line_shape.geom_type == 'MultiPolygon'):
                self._dynmic_elements_in_figure.append(self._simulation_ax.add_patch(plotting.patch_from_polygon(self._observed_finish_line_shape, linewidth=0, fc='r', ec='k', alpha=0.5)))

            # sensing wedge
            if not self._observed_sensing_wedge.is_empty:
                #ax.add_patch(plotting.patch_from_polygon(hidden_shape, color='orange', alpha=0.5))
                self._dynmic_elements_in_figure.append(self._simulation_ax.add_patch(plotting.patch_from_polygon(self._observed_sensing_wedge, linewidth=0, color='r', alpha=0.2)))




    def _WF_to_VF(self, WF_shape):
        """Transform a shape from world frame to vehicle frame."""

        return affinity.rotate(affinity.translate(WF_shape, -self.vehicle._state[0], -self.vehicle._state[1]), -self.vehicle._state[2], origin=(0,0), use_radians=True)

    def _VF_to_WF(self, VF_shape):
        """Transform a shape from vehicle frame to world frame."""

        return affinity.translate(affinity.rotate(VF_shape, self.vehicle._state[2], origin=(0,0), use_radians=True), self.vehicle._state[0], self.vehicle._state[1])

    def _update_vehicle_status(self):
        """Check if vehicle is safe/unsafe/reached goal (set the vehicle status and also return it)."""
        
        if self._finish_line_shape.contains(self._vehicle_shape):
            self.vehicle.status = VehicleStatus.FINISH
            return VehicleStatus.FINISH
        elif self._unobst_track_shape.contains(self._vehicle_shape):
            self.vehicle.status = VehicleStatus.SAFE
            return VehicleStatus.SAFE
        else:
            self.vehicle.status = VehicleStatus.UNSAFE
            return VehicleStatus.UNSAFE

    def _vehicle_state_changed_listener_initial(self):
        """Listener to vehicle state changes (called first)."""

        self._vehicle_shape = self._VF_to_WF(Vehicle.VF_shape)

    def _vehicle_state_changed_listener_final(self):
        """Listener to vehicle state changes (called at the end)."""

        self._update_vehicle_status()
        if self._is_simulation_fig_open:
            self._update_plot()

        # update history cache
        self._vehicle_state_history.append(self.vehicle.get_state())
        self._vehicle_shape_history.append(self._vehicle_shape)
        if self._latest_sensing_wedge is not None: # in initialization observation would be None, adds to history manually in ctor
            self._observation_wedges_history.append(self._latest_sensing_wedge)

    @staticmethod
    def _calc_track_shapes(env):
        """Calculate track and finish line shapes."""

        # calc track curve
        track_func_vectorized = vectorize(env.track_func)
        Simulation._X_track_curve = arange(env.X_track_range[0], env.X_track_range[1]+env.track_percision, env.track_percision)
        Simulation._Y_track_curve = track_func_vectorized(Simulation._X_track_curve)
        Simulation._track_length = env.track_arc_length(env.X_track_range[1])

        # calc track boundary
        X_track_top_boundary = []
        Y_track_top_boundary = []
        X_track_btm_boundary = []
        Y_track_btm_boundary = []

        for j in range(len(Simulation._X_track_curve)):
            x = Simulation._X_track_curve[j]
            y = Simulation._Y_track_curve[j]
            #y = scene.track_func(x)
            #Simulation._Y_track_curve.append(y)

            # find two track boundary points (on the normal of the curve tangent and in track_half_width distance from (x,y))
            # yy = m*(xx-x) + y
            # w**2 = (xx - x)**2 + (y - y)**2
            slope = env.track_drv_func(x)
            curvature = env.track_sec_drv_func(x)
            if slope == 0: # normal parllel to Y axis
                X_track_top_boundary.append(x)
                Y_track_top_boundary.append(y + env.track_padding)
                X_track_btm_boundary.append(x)
                Y_track_btm_boundary.append(y - env.track_padding)
            else:
                normal_slope = -1/slope
                bp1_x =  sqrt(env.track_padding**2/(normal_slope**2+1)) + x
                bp2_x = -sqrt(env.track_padding**2/(normal_slope**2+1)) + x 
                bp1_y = normal_slope*bp1_x - normal_slope*x + y
                bp2_y = normal_slope*bp2_x - normal_slope*x + y

                # decide which boundary point above/below the track
                if curvature <= 0 and slope < 0: # n \
                    X_track_top_boundary.append(bp1_x)
                    Y_track_top_boundary.append(bp1_y)
                    X_track_btm_boundary.append(bp2_x)
                    Y_track_btm_boundary.append(bp2_y)
                if curvature > 0 and slope < 0: # \ u
                    X_track_top_boundary.append(bp1_x)
                    Y_track_top_boundary.append(bp1_y)
                    X_track_btm_boundary.append(bp2_x)
                    Y_track_btm_boundary.append(bp2_y)
                if curvature <= 0 and slope > 0: # / n
                    X_track_top_boundary.append(bp2_x)
                    Y_track_top_boundary.append(bp2_y)
                    X_track_btm_boundary.append(bp1_x)
                    Y_track_btm_boundary.append(bp1_y)
                if curvature > 0 and slope > 0: # u /
                    X_track_top_boundary.append(bp2_x)
                    Y_track_top_boundary.append(bp2_y)
                    X_track_btm_boundary.append(bp1_x)
                    Y_track_btm_boundary.append(bp1_y)

        track_boundary = zip(X_track_top_boundary+X_track_btm_boundary[::-1],Y_track_top_boundary+Y_track_btm_boundary[::-1])
        Simulation._track_boundary = [[p[0],p[1]] for p in track_boundary]

        boundary_len = len(Simulation._track_boundary)        
        Simulation._btm_boundary_line = sg.LineString(Simulation._track_boundary[0:int(boundary_len/2)])
        Simulation._top_boundary_line = sg.LineString(Simulation._track_boundary[boundary_len-1:int(boundary_len/2)-1:-1])

        # calc track shape
        Simulation._track_shape = sg.Polygon(list(Simulation._track_boundary))
        
        # calc track finish line shape
        finish_line_width = round(env.finish_line_portion*len(X_track_top_boundary)) # in terms of percision units
        Simulation._finish_line_shape = sg.Polygon((p[0], p[1]) for p in zip(X_track_top_boundary[-finish_line_width:]+X_track_btm_boundary[-1:-1-finish_line_width:-1],\
                                                        Y_track_top_boundary[-finish_line_width:]+Y_track_btm_boundary[-1:-1-finish_line_width:-1]))

        # calc track start zone shape
        start_zone_width = round(env.start_zone_portion*len(X_track_top_boundary)) # in terms of percision units
        Simulation._start_zone_shape = sg.Polygon((p[0], p[1]) for p in zip(X_track_top_boundary[0:start_zone_width+1]+X_track_btm_boundary[start_zone_width::-1],\
                                                        Y_track_top_boundary[0:start_zone_width+1]+Y_track_btm_boundary[start_zone_width::-1]))

        # calc track shape without start and finish zones
        Simulation._track_shape_trimmed = Simulation._track_shape.difference(Simulation._start_zone_shape).difference(Simulation._finish_line_shape)