import numpy as np
import gym
# from IPython import embed
import deeprl_hw3.arm_env
from deeprl_hw3.ilqr import calc_ilqr_input
import copy
import time

env_name = 'TwoLinkArm-v0'
env = gym.make(env_name)
sim_env = gym.make(env_name)
initial_state = env.reset()
env.render()

total_cost = 0
num_steps = 0
tN=50
max_iter=100

x0 = copy.copy(env.state)

X, U, cost = calc_ilqr_input(env, sim_env, tN=tN, max_iter=max_iter, x0=x0)

for i in range(tN):

  print("Control u = {}, reward={}".format(str(U[i]), total_cost))
  x_next, cost_i, is_terminal, debug_info = env.step(U[i])
  env.render()

  total_cost += cost_i

  if is_terminal:
    break


