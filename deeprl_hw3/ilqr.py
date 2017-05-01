"""LQR, iLQR and MPC."""

import numpy as np
import scipy.linalg

from deeprl_hw3.controllers import DELTA, DT, approximate_A, approximate_B, simulate_dynamics, solve_riccati

def calc_ilqr_input(env, sim_env, tN=50, max_iter=1e6, x0=None):
  """Calculate the optimal control input for the given state.

  Parameters
  ----------
  env: gym.core.Env
    This is the true environment you will execute the computed
    commands on. Use this environment to get the Q and R values as
    well as the state.
  sim_env: gym.core.Env
    A copy of the env class. Use this to simulate the dynamics when
    doing finite differences.
  tN: number of control steps you are going to execute
  max_itr: max iterations for optmization

  Returns
  -------
  U: np.array
    The SEQUENCE of commands to execute. The size should be (tN, #parameters)
  """
  u = np.zeros((env.action_space.shape[0], tN))
  x = np.zeros((env.observation_space.shape[0], tN + 1))
  for i in range(tN):
    x[:, i] = env.state

  for i in range(int(max_iter)):
    print("Iter no.", i)

    old_x = x.copy()

    # Forward pass
    sim_env.state = x0.copy()
    for i in range(tN):
      A = approximate_A(sim_env, x[:, i], u[:, i])
      B = approximate_B(sim_env, x[:, i], u[:, i])
      c = simulate_dynamics(sim_env, x[:, i], u[:, i]) - A.dot(x[:, i]) - B.dot(u[:, i]) - env.goal + A.dot(env.goal)

      P1, P2 = solve_riccati(A, B, c, env.Q, env.R)
      error = x[:, i] - env.goal
      u[:, i] = -np.linalg.solve(env.R + B.T.dot(P1).dot(B), 0.5 * P2.dot(B) + B.T.dot(P1).dot(c) + B.T.dot(P1).dot(A).dot(error))

      sim_env.state = x[:, i].copy()
      sim_env.render()
      next_state, reward, is_terminal, debug_info = sim_env.step(u[:, i])
      x[:, i + 1] = next_state

    if np.linalg.norm(x - old_x) < 1e-2:
      break

  return u.T
