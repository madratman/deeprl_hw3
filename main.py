import numpy as np
import gym
from IPython import embed
import deeprl_hw3.arm_env
from deeprl_hw3.controllers import calc_lqr_input
import copy
import time

env = gym.make("TwoLinkArm-v0")
initial_state = env.reset()
env.render()
time.sleep(1)

dt = 1e-5
total_reward = 0
num_steps = 0
while True:
  u = calc_lqr_input(env, copy.deepcopy(env))
  u = np.array(u)
  u = np.reshape(u, env.action_space.shape[0])
  nextstate, reward, is_terminal, debug_info = env.step(u,dt)
  env.render()

  total_reward += reward
  num_steps += 1

  if is_terminal:
  	break

  time.sleep(1)