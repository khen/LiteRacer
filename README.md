# Quick Start
## What it LiteRacer?

`LiteRacer` is a lightweight simulator of an autonmous car driving on a racetrack. 
The basic simulation scenario includes an autonmous car with a lidar-like sensor and a Neural Network (NN)-based controller, pretrained to steer the car to the endzone of the sinuous track without collision, despite environment variability.
The simulator allows the user to easily adjust the car and track parameters, to place obstacles on the track, to configure the controller, motion, and observation models, and to determine the formal specification of a successful run.

`LiteRacer` was designed to support the development of formal verification techniques for autonomous Cyber-Physical Systems (CPSs) with NN-based components, such as controllers or perception models. 
With `LiteRacer`, the user can then easily simulate system runs in a variety of pre-selected or randomly generated scenarios, to test whether a specification is satisfied. By such, the simulator can be used for:

1. Component training/training-algorithm evaluation.

2. System/component testing: to evaluate whether a system with the specified componenets satisfies robustly and consistently the desired specification, under changing conditions.

3. Development and benchmarking of falsification and verification algorithms. 

## Citation information
If you benefit from `LiteRacer` in your work, please make sure to properly cite it, using the following format:
```bibtex
@inproceedings{Elimelech24LiteRacer,
author = {Elimelech, Khen and Lahijanian, Morteza and Kavraki, Lydia E. and Vardi, Moshe Y.},
title = {{L}ite{R}acer: a lightweight autonomous vehicle simulator for benchmarking and development of formal verification techniques},
booktitle = {Workshop on Software Challenges in Formal Methods for Robotics (FMR), in conjunction with {IEEE} International Conference on Robotics and Automation ({ICRA})},
year = {2024},
month = {05},
location = {Yokohama, Japan},
url	= {https://github.com/khen/LiteRacer},
}
```

## Installation and dependencies
The `literacer` package is purely Python-based. This means that the simulator itself is written in Python, and its operation is done through low-hassle Python (or command line) scripts.

Beyond the standard python packages (`numpy`, `scipy`, `matplotlib`), `literacer` depends only on the following packages:

- `shapely` (for geometric processing)
    - <https://anaconda.org/conda-forge/shapely>
- `stable-baselines3` (for interfacing with the neural controller):
	- <https://anaconda.org/conda-forge/stable-baselines3>

> At the moment, `LiteRacer` is not provided as an installable package, but only as the downloadable module `literacer`. To use, simply download the `literacer\` folder into a desired location and install the aformentioned dependencies (we recommend doing so in a `conda` or `pip` virtual environment).

### Troubleshooting
> The pretrained model was created using stables-baslines3 version `1.1.0`. If loading the model using the latest version of `stable_baseline3` fails, try downgrading to version `1.1.0` instead.

  
## Using `LiteRacer`
There are two possible work schemes:

### Command line interface
The simpler one is through the module's command line interface. To work in these sceheme simply download the `literacer` module (folder) and the configuration file `simulation_config.py` into an accessible location on your machine.
Then, from the location containing the module folder and the configuration file, you may simply run the following command line script, to initiate the simulation:
```
python -m literacer 
```
Control of the scenario, vehicle, and visualizer properties is done by modifying `simulation_config.py`. We recommend examining the available configuration parameters before starting your project.
Make sure the package dependencies are installed in your environment.

You may also run the command with the following flags:
```
python -m literacer --runs N --vis on/off --config FOLDER_PATH
```
These corrspond to the desired number of simulated runs to initiate sequentially (if the visualztion is on, a new run will start once you close the visualizer window), whether the visualization should be `on` or `off`, and an alternative location for the configuration file (the default location is the current working directory), respectively.

### Access through code
Alternatively, one can invoke the simulator through Python code.
This allows one to automate, analyze, and interrupt the run of the simulation.

Again, start by downloading the `literacer` module (folder) and the configuration file `simulation_config.py` into an accessible location on your machine.
Then, create in a new Python script with a `Simulation` object. In that case, `simulation_config.py` should be imported as a module and passed to the simulation object upon its creation, to set its properties. A simple script should look like this:
```python
import simulation_config
from literacer.components.simulation import Simulation
from literacer.utils.enums import VehicleStatus

# create a simulation object and open the visualizer
simulation = Simulation(simulation_config)
simulation.open_visualizer()

# run the simulation
simulation.vehicle.run()

# check vehicle status
if simulation.vehicle.status == VehicleStatus.FINISH:
	print("Success.")
else:
	print("Failure."")
```
Operating the simulation is done by invoking the public interface of the `Simulation` object, or the `Vehicle` object it contains. 
Through this interface one can, for example, limit the number of control steps the vehicle can take at a time, add obstacles dynamically at run time, or check the vehicle status.


### Using `LiteRacer` for testing and verification
Demonstration scripts of using `LiteRacer` to evaluate and benchmark falsification (black-box verification) algorithms are provided in `demo\falsification\`. This includes our state-of-the-art approach: Meta-Planning [1].

[![arXiv metaplanning](https://img.shields.io/badge/arXiv-2412.17992-b31b1b.svg)](https://arxiv.org/abs/2412.17992)

```
[1] "Falsification of Autonomous Systems in Rich Environments," K. Elimelech, M. Lahijanian, L. Kavraki, and M. Vardi, arXiv:2412.17992, 2024. et al.
```

### Useful tips
- The configuration parameter `vehicle_config.delta_t` sets the speed of the simulation, in expense of smoothness of motion. 
- When changing the configuration variable `vehicle_config.sensing_freq` to `ON_REQUEST`, the simulation will only invoke a new observation once every control loop, and not continuously. This should make the simulation run faster, yet, in most cases, should not affect the functionality (though it might cause the observation smoothness in the visualizer). Changing this option can be useful, e.g., when running many simulations with the visualizier turned off.
- New obstacles may be added either in a location in relation to the track, or by specifying an absolute "XY location" (in "World Frame"). Examine the corresponding function signatures in the `Simulation` class. The predefined obstacles specified in `simulation_config` are added in relation to the track.


## Advanced functionality
### Changing controller
`literacer` comes with two pre-trained neural controllers, available in `literacer/resources/controller_models/`. Changing between the models can be done through the configuration (`vehicle_config.controller_model_path`).
We currently only support `stable-baselines3` models. If you wish to load a different model, you will have to create a new controller class, by inheriting `literacer.componenet.controller` and overriding `_load_controller_model()`. This new controller should then be specified in the connfiguration (`vehicle_config.controllerClass`).

### Training a new controller
By integrating with `gym` (<https://anaconda.org/conda-forge/gym>), `literacer` also allows the user to conveniently train the system components (such as the controller of perception model). An example of controller training is provided in `demo\controller_training\`.

> When training a new controller model, make sure the configuration property `vehicle_config.controllerClass` points to `None`, so the simulation avoids loading an existing model.

## Release history
- 17/07/2025: v0.9.0 (pre-release)
- 01/04/2024: Beta version