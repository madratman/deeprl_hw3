import numpy as np
import gym
import deeprl_hw3.arm_env
from deeprl_hw3.controllers import calc_lqr_input
import copy
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

env_name = 'TwoLinkArm-v0'
env = gym.make(env_name)
sim_env = gym.make(env_name)
initial_state = env.reset()
env.render()

total_reward = 0
num_steps = 0

Q = []
U = []
traj_iter = 0

while True:
  traj_iter += 1
  if num_steps >= 0:
    u = calc_lqr_input(env, sim_env, np.array((0., 0.)))
  else:
    u = calc_lqr_input(env, sim_env, prev_u)

  prev_u = u
  print("Control u = {}, num_steps={}, reward={}".format(str(u), num_steps, total_reward))
  nextstate, reward, is_terminal, debug_info = env.step(u)
  env.render()

  total_reward += reward
  num_steps += 1

  Q.append(env.state)
  U.append(u)

  if is_terminal:
  	break

Q = np.array(Q)
U = np.array(U)

plot_clipped_u = 'limited' in env_name.split('-')
# num_subplots = 4 if plot_clipped_u else 3

plt.subplot(411)
plt.xlabel('t')
plt.ylabel('Control u')
plt.title('LQR Solution for ' + env_name)
blue_patch = mpatches.Patch(color='blue', label='Control 0')
red_patch = mpatches.Patch(color='red', label='Control 1')
plt.legend(handles=[red_patch, blue_patch])
plt.plot(range(traj_iter), U[:,0], 'b', range(traj_iter), U[:,1], 'r')

plt.subplot(412)
plt.xlabel('t')
plt.ylabel('q/position')
blue_patch = mpatches.Patch(color='blue', label='q 0')
red_patch = mpatches.Patch(color='red', label='q 1')
plt.legend(handles=[red_patch, blue_patch])
plt.plot(range(traj_iter), Q[:,0], 'b', range(traj_iter), Q[:,1], 'r')

plt.subplot(413)
plt.xlabel('t')
plt.ylabel('q_dot/velocity')
blue_patch = mpatches.Patch(color='blue', label='q_dot 0')
red_patch = mpatches.Patch(color='red', label='q_dot 1')
plt.legend(handles=[red_patch, blue_patch])
plt.plot(range(traj_iter), Q[:,2], 'b', range(traj_iter), Q[:,3], 'r')

if plot_clipped_u:
  plt.subplot(414)
  plt.xlabel('t')
  plt.ylabel('Clipped Control u')
  blue_patch = mpatches.Patch(color='blue', label='Clipped control 0')
  red_patch = mpatches.Patch(color='red', label='Clipped control 1')
  plt.legend(handles=[red_patch, blue_patch])
  U_clipped = np.clip(U, env.action_space.low, env.action_space.high)
  plt.plot(range(traj_iter), U_clipped[:,0], 'b', range(traj_iter), U_clipped[:,1], 'r')

plt.show()
