import glob
import os

import sys
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
from stable_baselines.common.evaluation import evaluate_policy
import carla
import gym
from gym import spaces
from carla import ColorConverter as cc
import random
import time
import numpy as np
import cv2
import pygame
import weakref
from datetime import datetime
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.policies import CnnPolicy,CnnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import results_plotter
#from stable_baselines.bench import Monitor
from monitor import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines import TRPO,ACER,PPO2,ACKTR,DDPG,A2C,SAC
import settings
from CarEnv import CarEnv
import argparse
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "" 

parser = argparse.ArgumentParser(description='Short sample app')

parser.add_argument('-m','--model',default='trpo')
parser.add_argument('-e','--eval')
parser.add_argument("-a",'--action',default='discrete')
args = parser.parse_args()



class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True

def evaluate(model,env):
    env.save_video = True
    # Enjoy trained agent
    obs = env.reset()
    eps = 10
    rewards_list = []
    distances = []
    final_results = []
    cnt = 0
    while True:
        env.render(mode='video')
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        rewards_list.append(rewards)
        if done:
            result = {'reward':np.mean(rewards_list),'distance':info['distance']}
            final_results.append(result)
            env.reset()
            cnt += 1
            print(result)
        if cnt > eps:
            break
    env.close()
    return final_results

def train(model,env,log_dir):
    print("-----Start training------")
    # Create the callback: check every 10
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    # Train the agent
    time_steps = 1e6
    model.learn(total_timesteps=int(time_steps), callback=callback)
    env.close()

if __name__ == '__main__':
    if not args.eval:
        now = datetime.now()
        now = now.strftime("%d%m%Y_%H%M%S")
        out_dir = '{}_{}_{}_out'.format(now[:-4],args.model,args.action)
        os.makedirs(out_dir, exist_ok=True)
        env = CarEnv(out_dir,n_stacks=5,a_space_type=args.action)
        env.next_weather()
        env = Monitor(env, out_dir)

        print("==========Creating model------------------")
        policy = CnnPolicy
        if args.model == 'trpo':
            model = TRPO(policy, env, verbose=1, timesteps_per_batch=64,tensorboard_log=out_dir)
        elif args.model == 'acer':
            model = ACER(policy, env, verbose=1, n_steps=64,tensorboard_log=out_dir)
        elif args.model == 'ppo':
            model = PPO2(policy,env,verbose=1,n_steps=64,tensorboard_log=out_dir)
        elif args.model == 'acktr':
            model = ACKTR(policy,env,n_steps=4, verbose=1,tensorboard_log=out_dir)
        elif args.model == 'ddpg':
            model = DDPG(policy,env,verbose=1,tensorboard_log=out_dir)
        elif args.model == 'a2c':
            model = A2C(policy,env,n_steps=64, verbose=1,tensorboard_log=out_dir)
        elif args.model == 'sac':
            model = SAC("CnnPolicy",env)
        train(model,env,out_dir)
    else:
    #results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "rl")
        path = '{}/best_model.zip'.format(args.eval)
        env = CarEnv(args.eval,cam_idx_list=(0,3,4))
        env.next_weather()
        #env = Monitor(env, args.eval)
        #print(env.num_envs)
        if args.model == 'trpo':
            model = TRPO.load(path)
        elif args.model == 'acer':
            model = ACER.load(path)
        elif args.model == 'ppo':
            model = PPO2.load(path)
        elif args.model == 'acktr':
            model = ACKTR.load(path)
        elif args.model == 'ddpg':
            model = DDPG.load(path)
        elif args.model == 'a2c':
            model = A2C.load(path)
        elif args.model == 'sac':
            model = SAC.load(path)
        #mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5,return_episode_rewards=True)
        #eps_rewards, eps_len = evaluate_policy(model, env, n_eval_episodes=5,return_episode_rewards=True)
        # print(eps_rewards)
        # print(eps_len)
        # print(np.mean(eps_rewards))
        #print("Mean reward = {}","Std reward = {}".format(np.mean(eps),std_reward))
        rs = evaluate(model,env)
        with open("{}/result.txt".format(args.eval), 'w') as f:
            for item in rs:
                f.write("%s\n" % item)
        # f = open("{}/result.txt".format(args.eval), "w")
        # f.write(rs)
        # f.close()
    