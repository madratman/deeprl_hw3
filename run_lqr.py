import numpy as np
import gym
# from IPython import embed
import deeprl_hw3.arm_env
from deeprl_hw3.controllers import calc_lqr_input
import copy
import time

env_name = 'TwoLinkArm-limited-torque-v0'
env = gym.make(env_name)
sim_env = gym.make(env_name)
initial_state = env.reset()
env.render()

total_reward = 0
num_steps = 0

while True:
  if num_steps >= 0:
    u = calc_lqr_input(env, sim_env)
  else:
    u = calc_lqr_input(env, sim_env, prev_u)
  u = np.array(u)
  u = np.reshape(u, env.action_space.shape[0])
  prev_u = u
  print("Control u = {}, num_steps={}, reward={}".format(str(u), num_steps, total_reward))
  nextstate, reward, is_terminal, debug_info = env.step(u)
  env.render()

  total_reward += reward
  num_steps += 1

  if is_terminal:
  	break