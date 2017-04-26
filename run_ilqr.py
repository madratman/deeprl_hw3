import numpy as np
import gym
from IPython import embed
import deeprl_hw3.arm_env
from deeprl_hw3.ilqr import calc_ilqr_input
import copy
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

env_name = 'TwoLinkArm-v1'
env = gym.make(env_name)
sim_env = gym.make(env_name)
initial_state = env.reset()

total_cost = 0
num_steps = 0
tN = 50
max_iter = 100

x0 = copy.copy(env.state)

X, U, cost, list_of_costs = calc_ilqr_input(env, sim_env, tN=tN, max_iter=max_iter, x0=x0)

R = []

for i in range(tN):

  print("Control u = {}, reward={}".format(str(U[i]), total_cost))
  x_next, cost_i, is_terminal, debug_info = env.step(U[i])
  env.render()

  total_cost += cost_i
  R.append(cost_i)

  if is_terminal:
    break

R = np.array(R)


plt.subplot(411)
plt.title('iLQR Solution for ' + env_name)
plt.xlabel('iLQR terations')
plt.ylabel('Total Cost')
blue_patch = mpatches.Patch(color='blue', label='Total cost per iLQR iteration')
plt.legend(handles=[blue_patch])
plt.plot(range(len(list_of_costs)), np.array(list_of_costs), 'b')

# This subplot just plots the final executed cumulative total cost
# plt.subplot(411)
# plt.xlabel('t')
# plt.ylabel('Cost, upon execution')
# blue_patch = mpatches.Patch(color='blue', label='Total cumulative cost')
# plt.legend(handles=[blue_patch])
# plt.plot(range(tN), R, 'b')

plt.subplot(412)
plt.xlabel('t')
plt.ylabel('Control u')
blue_patch = mpatches.Patch(color='blue', label='Control 0')
red_patch = mpatches.Patch(color='red', label='Control 1')
plt.legend(handles=[red_patch, blue_patch])
plt.plot(range(tN), U[:,0], 'b', range(tN), U[:,1], 'r')

plt.subplot(413)
plt.xlabel('t')
plt.ylabel('q/position')
blue_patch = mpatches.Patch(color='blue', label='q 0')
red_patch = mpatches.Patch(color='red', label='q 1')
plt.legend(handles=[red_patch, blue_patch])
plt.plot(range(tN), X[:,0], 'b', range(tN), X[:,1], 'r')

plt.subplot(414)
plt.xlabel('t')
plt.ylabel('q_dot/velocity')
blue_patch = mpatches.Patch(color='blue', label='q_dot 0')
red_patch = mpatches.Patch(color='red', label='q_dot 1')
plt.legend(handles=[red_patch, blue_patch])
plt.plot(range(tN), X[:,2], 'b', range(tN), X[:,3], 'r')

plt.show()

