"""Controller."""

from math import pi
import numpy as np
from PIL import ImageOps


class Controller():
    """Vehicle controller."""

    _controllerModel = None
    """NN contoller model, inputing observation and steering value and outputing a control action."""
    
    def __init__(self, vehicle_config):
        """Initialize controller as specified in the configuration."""
        
        self.observation_image_size = vehicle_config.observation_image_size
        self.steering_angle_range = vehicle_config.steering_angle_range
        
        # initialize controller if not in controller training mode
        if Controller._controllerModel is None:
            self._load_controller_model(vehicle_config)

    def get_control_action(self, observation_image, steering):
        """Return a control action (speed, steering speed) given a sensor observation and steering value."""

        # call NN model
        action = Controller._controllerModel.predict(self.arrange_controller_input(observation_image, steering), deterministic=True)
        # returns a list of actions (for a list of observations), so take only the first element
        return action[0][0]
    
    def arrange_controller_input(self, observation_image, steering):
        """Mitigate sensor output and control model input formats.
           Arrange raw observation and steering value in input format expected by the NN controller model."""

        observation_array = self._compress_sensor_image_to_array(observation_image)
        # number of channels
        observation_array_augmented = np.expand_dims(observation_array, axis=0)
        steering_array = np.array([steering], dtype=np.float32)

        # model expecting a batch/list of observation of the specified shape, so must augment (only for non-scalar inputs)
        observation_array_augmented = np.expand_dims(observation_array_augmented, axis=0)
        steering_array = np.expand_dims(steering_array, axis=0)

        return {'scan': observation_array_augmented, 'steering': steering_array}

    def _compress_sensor_image_to_array(self, observation_image):
        """Downsample observation image and save in array."""

        observation_image = ImageOps.grayscale(observation_image)
        observation_image = observation_image.resize((self.observation_image_size[1], self.observation_image_size[0])) # width, height
        observation_array = np.asarray(observation_image)
        return observation_array

    def _load_controller_model(self, vehicle_config):
        """Set NN controller model as specified in the configuration."""

        try:
            # load stable_baslines model according to it type (training algorithm)
            import stable_baselines3 
            model_type = getattr(stable_baselines3, vehicle_config.controller_model_type)
            load_func = getattr(model_type, 'load')
            import os 
            if os.path.isabs(vehicle_config.controller_model_path) == True:
                # given an absolute path to controller model
                Controller._controllerModel = load_func(vehicle_config.controller_model_path)
            else:
                # given relative path, calc in relation to package location
                parent_dir_of_current_file = "\\".join(os.path.realpath(__file__).split("\\")[0:-2])
                Controller._controllerModel = load_func(parent_dir_of_current_file+"\\"+vehicle_config.controller_model_path)
        except Exception as e:
            raise Warning('Unable to load vehicle controller model.')