"""LQR, iLQR and MPC."""

import numpy as np
import scipy.linalg
import gym
# from IPython import embed
import copy

DELTA = 1e-5
DT = 1e-5

def simulate_dynamics(env, x, u, dt=DT):
    """Step simulator to see how state changes.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are trying to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. When approximating A you will need to perturb
      this.
    u: np.array
      The command to test. When approximating B you will need to
      perturb this.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    xdot: np.array
      This is the **CHANGE** in x. i.e. (x[1] - x[0]) / dt
      If you return x you will need to solve a different equation in
      your LQR controller.
    """
    x_orig = copy.copy(x)
    env.state = x_orig
    xnew, _, _, _, = env._step(u, dt)
    xdot = (xnew - x)/dt
    return xdot

def approximate_A(env, x, u, delta=DELTA, dt=DT):
    """Approximate A matrix using finite differences.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. You will need to perturb this.
    u: np.array
      The command to test.
    delta: float
      How much to perturb the state by.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    A: np.array
      The A matrix for the dynamics at state x and command u.
    """
    x_orig = copy.copy(x)

    # initialize matrix A
    A = np.zeros([env.observation_space.shape[0],env.observation_space.shape[0]])

    for i in range(env.observation_space.shape[0]):

      x_perturbed = copy.copy(x_orig)
      x_perturbed[i] += delta
      x_dot1 = simulate_dynamics(env, x_perturbed, u, dt=DT)

      x_perturbed = copy.copy(x_orig)
      x_perturbed[i] -= delta
      x_dot2 = simulate_dynamics(env, x_perturbed, u, dt=DT)

      delta_x = (x_dot1 - x_dot2)/(2*delta)
      A[i,:] = delta_x
      # comment
    return A

def approximate_B(env, x, u, delta=DELTA, dt=DT):
    """Approximate B matrix using finite differences.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test.
    u: np.array
      The command to test. You will ned to perturb this.
    delta: float
      How much to perturb the state by.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    B: np.array
      The B matrix for the dynamics at state x and command u.
    """
    u_orig = copy.copy(u)

    # initialize matrix B
    B = np.zeros([env.observation_space.shape[0],env.action_space.shape[0]])

    for i in range(env.action_space.shape[0]):

      u_perturbed = copy.copy(u_orig)
      u_perturbed[i] += delta
      x_dot1 = simulate_dynamics(env, x, u_perturbed, dt=DT)

      u_perturbed = copy.copy(u_orig)
      u_perturbed[i] -= delta
      x_dot2 = simulate_dynamics(env, x, u_perturbed, dt=DT)

      delta_x = (x_dot1-x_dot2)/(2*delta)
      B[:,i] = delta_x

    return B

def calc_lqr_input(env, sim_env, prev_u=None):
    """Calculate the optimal control input for the given state.

    If you are following the API and simulate dynamics is returning
    xdot, then you should use the scipy.linalg.solve_continuous_are
    function to solve the Ricatti equations.

    Parameters
    ----------
    env: gym.core.Env
      This is the true environment you will execute the computed
      commands on. Use this environment to get the Q and R values as
      well as the state.
    sim_env: gym.core.Env
      A copy of the env class. Use this to simulate the dynamics when
      doing finite differences.

    Returns
    -------
    u: np.array
      The command to execute at this point.
    """
    
    # get the values for the matrices
    x = env.state
    u = np.random.rand(2)
    A = approximate_A(sim_env, x, u, delta=DELTA, dt=DT)
    B = approximate_B(sim_env, x, u, delta=DELTA, dt=DT)
    Q = env.Q
    R = env.R
    # sdgasf
    # Solve ARE equation, continuous time
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))

    # Compute the LQR gain
    K = np.matrix( scipy.linalg.inv(R).dot(B.T.dot(X)))
    # u = -K.dot(x - env.goal)
    u = -K.dot(x-env.goal)

    print x
    print u
    print A
    print B
    print Q
    print R

    return u