### To be able to call files from parent folder
import os, sys
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
###
 
from os.path import isfile
from datetime import datetime, date
from stable_baselines3 import SAC, A2C, PPO
from stable_baselines3.common import logger, monitor, noise
from stable_baselines3.common.callbacks import EvalCallback
from controllerGymEnv import ControllerGymEnv
from resources.config import env_config

# create timestamp for file
today = date.today()
now = datetime.now()
current_time = now.strftime("%H%M%S")
output_path = "./model_"+str(today)+"_"+str(current_time)+"/"
#output_path = "./models/"

# load pretrained model or create a new one if doesn't exist
# NOTE: CONTINUED LEARNING DOES NOT WORK WELL WITH SAC (stable_baslines bug)
vehicle_env_monitor = monitor.Monitor(ControllerGymEnv(verbose=1))

if isfile(output_path+"best_model.zip"):
    print("Loading pretrained model...")
    model = SAC.load(output_path, env=vehicle_env_monitor, reset_num_timesteps=False)
else:
    print("Creating a new model...")
    model = SAC("MultiInputPolicy", vehicle_env_monitor)

# set model config for learning # TODO: see if config can be saved and loaded with the model itself, instead
model.verbose = 1
# model.buffer_size = 100000
# model.action_noise = noise.NormalActionNoise(array([0.225,0]), array([0.2,0.2]))
# model.learning_rate = 0.005
# model.learning_starts = 1

# Separate evaluation env
eval_env = monitor.Monitor(ControllerGymEnv())
eval_callback = EvalCallback(eval_env, best_model_save_path=output_path,
                             eval_freq=5000, n_eval_episodes = 40, deterministic=False)

model.learn(total_timesteps=10000000, callback=eval_callback)