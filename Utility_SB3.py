# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 01:01:47 2023

@author: ericl
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

from stable_baselines3 import A2C
# from stable_baselines3.common.env_checker import check_env
# from stable_baselines3.common.monitor import Monitor, ResultsWriter
# from stable_baselines3.common.results_plotter import plot_results, plot_curves
# from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

import Utility
# from SAWRC_Simulator import Simulator


# In[callback functions]
def make_env(
        env_id: str, 
        hyperparameters_env: dict, 
        rank: int, 
        seed: int = 0, 
        ):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param hyperparameters_env: env setup parameters 
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = gym.make(
            env_id, 
            pygame_visualization=0, 
            data_autosave=0, 
            **hyperparameters_env, 
            )
        env.reset()
        return env
    return _init


def make_vectorized_env(
        env_id: str, 
        hyperparameters_env: dict, 
        num_env: int, 
        wrapper: str = 'SubprocVecEnv', 
        ): 
    """
    Utility function for making vectorized envs

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    if wrapper == 'DummyVecEnv': 
        env_vec = DummyVecEnv([make_env(env_id, hyperparameters_env, i) for i in range(num_env)])
    elif wrapper == 'SubprocVecEnv': 
        env_vec = SubprocVecEnv([make_env(env_id, hyperparameters_env, i) for i in range(num_env)])
    print(num_env, 'envs generated')
    
    return env_vec 


def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float):
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value
    return func


def evaluate_trained_model(
        env_id: str, 
        hyperparameters_env: dict, 
        saved_model_name: str, 
        total_timesteps: int, 
        progress_bar: bool = False, 
        print_policy_info: bool = False, 
        ): 
    """
    Evaluate trained model and plot rewards
    """
    env = gym.make(
        env_id, 
        pygame_visualization=0, 
        data_autosave=1, 
        **hyperparameters_env, 
        )
    model_trained = A2C.load(saved_model_name, env=env)
    if print_policy_info: 
        print(model_trained.policy)
    env_unwrapped = model_trained.get_env().envs[0].unwrapped
    obs, info = env_unwrapped.reset()
    cumreward_list = [0]
    if progress_bar: 
        iterable = tqdm(range(total_timesteps)) 
    else:
        iterable = range(total_timesteps)
    for ii in iterable:
        action, _states = model_trained.predict(obs, deterministic=True)
        obs, rewards, dones, truncated, info = env_unwrapped.step(action)
        cumreward_list.append(env_unwrapped.cumreward)
    env_unwrapped.close()
    
    # plot cumulative reward 
    fig, ax = plt.subplots()
    fig.dpi = 200 
    ax.plot(cumreward_list)
    ax.set_xlabel('time step') 
    ax.set_title('cumulative reward')
    
    simulator = env_unwrapped.simulator
    cd_dir = simulator.saved_construction_data_dir
    if cd_dir != None: 
        cd = Utility.load_data(*cd_dir)
        simulator.characterize_construction(cd, tag=saved_model_name)
        plt.show()
        
    return simulator, cd 


class SaveModelCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(
            self, 
            env_id: str, 
            hyperparameters_env: dict, 
            num_env: int, 
            check_freq: int, 
            model_name_prefix: str, 
            evaluation_time: int, 
            ):
        super(SaveModelCallback, self).__init__(verbose=1)
        self.env_id = env_id 
        self.hyperparameters_env = hyperparameters_env
        self.check_freq = check_freq
        self.model_name_prefix = model_name_prefix 
        self.evaluation_time = evaluation_time 
        self.num_env = num_env 
        self.saved_model_name_list = []

    def _on_step(self) -> bool:
        total_time_elapsed = self.n_calls * self.num_env
        if total_time_elapsed % self.check_freq == 0: 
            # save model 
            model_name_suffix = 'TT' + str(total_time_elapsed) 
            saved_model_name = self.model_name_prefix + model_name_suffix
            self.model.save(saved_model_name) 
            self.saved_model_name_list.append(saved_model_name)
            # evaluate saved model 
            print('evaluating:', saved_model_name)
            evaluate_trained_model(self.env_id, self.hyperparameters_env, saved_model_name, self.evaluation_time) 

        return True
    
    
# In[hyperparameter management]
def get_hyperparameters(hp_model_version, hp_env_version): 
    # get environment hyperparameters
    if hp_env_version == 'v0': 
        hp_env = {
            'observation_space_type': 'partial_v1', 
            'action_space_type': 'continuous_v1', 
            'reward_function': 'W1_reward_v1', 
            }
        env_id = 'sawrc_gym:Leveling0-v2' 
        num_env = 1 
    elif hp_env_version == 'v1': 
        hp_env = {
            'observation_space_type': 'full_v1', 
            'action_space_type': 'continuous_v1', 
            'reward_function': 'W1_reward_v4', 
            }
        env_id = 'sawrc_gym:Leveling0-v2' 
        num_env = 4 
    elif hp_env_version == 'v2': 
        hp_env = {
            'reward_function': 'W1_reward_v1', 
            }
        env_id = 'saw_gym:Moving0-v0'
        num_env = 1
    
    # get model hyperparameters
    if hp_model_version == 'test': 
        hp_model = {
            'n_steps': 5, 
            'learning_rate': linear_schedule(0.001), 
            }
    elif hp_model_version == 'v0': 
        hp_model = {
            'gamma': 1, 
            'learning_rate': linear_schedule(0.00075), 
            'n_steps': 100,
            'policy_kwargs': dict(net_arch=[64, 128, 64]),
            }
    elif hp_model_version == 'v1': 
        # tuned parameters of MountainCarContinuous-v0 from rl-zoo3
        # url: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/a2c.yml 
        """
          n_envs: 4
          n_steps: 100
          n_timesteps: !!float 1e5
          use_sde: True
          sde_sample_freq: 16
          policy_kwargs: "dict(log_std_init=0.0, ortho_init=False)"
        """
        batch_size = 4 * 100 
        hp_model = {
            'n_steps': int(batch_size/num_env), 
            'use_sde': True,
            'sde_sample_freq': 16,
            'policy_kwargs': dict(log_std_init=0.0, ortho_init=False),
            }
    elif hp_model_version == 'v2': 
        # tuned parameters of BipedalWalker-v3 from rl-zoo3 
        # url: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/a2c.yml 
        """
          n_envs: 16
          n_timesteps: !!float 5e6
          n_steps: 8
          gae_lambda: 0.9
          vf_coef: 0.4
          learning_rate: lin_0.00096
          use_sde: True
          policy_kwargs: "dict(log_std_init=-2, ortho_init=False)"
        """
        batch_size = 8 * 16 
        hp_model = {
            'n_steps': int(batch_size/num_env), 
            'gae_lambda': 0.9, 
            'vf_coef': 0.4, 
            'learning_rate': linear_schedule(0.00096), 
            'use_sde': True,
            'policy_kwargs': dict(log_std_init=-2, ortho_init=False), 
            }
    elif hp_model_version == 'v3': 
        hp_model = {
            'n_steps': 100, 
            }
    elif hp_model_version == 'v4': 
        hp_model = {
            'n_steps': 10, 
            }
    elif hp_model_version == 'v5': 
        hp_model = {
            'n_steps': 100, 
            'gamma': 1, 
            }
    elif hp_model_version == 'v6': 
        hp_model = {
            'n_steps': 10, 
            'gamma': 1, 
            }
    elif hp_model_version == 'v7': 
        hp_model = {
            'n_steps': 100, 
            'gamma': 1, 
            'learning_rate': 0.001, 
            }
        
    return hp_env, env_id, num_env, hp_model
    
    
# In[]
if __name__ == "__main__": 
    hyperparameters_env = {
        'observation_space_type': 'partial_v1', 
        'action_space_type': 'continuous_v1', 
        'reward_function': 'W1_reward_v3', 
        }
    env_id = 'sawrc_gym:Leveling0-v2' 
    saved_model_name = 'Leveling0-v1-a2c_HPv3_2023-10-07-06-47-34_TT16000'
    evaluation_time = 100
    
    simulator, cd = evaluate_trained_model(
        env_id=env_id, 
        hyperparameters_env=hyperparameters_env, 
        saved_model_name=saved_model_name, 
        total_timesteps=evaluation_time, 
        progress_bar=True, 
        )