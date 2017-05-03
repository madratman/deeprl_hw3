"""LQR, iLQR and MPC."""

import numpy as np
import scipy.linalg
from deeprl_hw3.controllers import DELTA, DT
import copy
from IPython import embed

LAMB_FACTOR=10
EPS_CONVERGE=1e-10
LAMB_MAX=1000

# reference for this code: StudyWolf blog, found at https://studywolf.wordpress.com/2016/02/03/the-iterative-linear-quadratic-regulator-method/


def simulate_dynamics_cont(env, x, u, dt=DT):
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


def simulate_dynamics_next(env, x, u, dt=None):
  """Step simulator to see how state changes.

  Parameters
  ----------
  env: gym.core.Env
    The environment you are try to control. In this homework the 2
    link arm.
  x: np.array
    The state to test. When approximating A you will need to perturb
    this.
  u: np.array
    The command to test. When approximating B you will need to
    perturb this.

  Returns
  -------
  next_x: np.array
  """

  x_orig = copy.copy(x)
  env.state = x_orig
  xnew, _, _, _, = env._step(u)

  return xnew


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
      x_dot1 = simulate_dynamics_cont(env, x_perturbed, u, dt=DT)

      x_perturbed = copy.copy(x_orig)
      x_perturbed[i] -= delta
      x_dot2 = simulate_dynamics_cont(env, x_perturbed, u, dt=DT)

      delta_x = (x_dot1 - x_dot2)/(2*delta)
      A[:,i] = delta_x
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
      x_dot1 = simulate_dynamics_cont(env, x, u_perturbed, dt=DT)

      u_perturbed = copy.copy(u_orig)
      u_perturbed[i] -= delta
      x_dot2 = simulate_dynamics_cont(env, x, u_perturbed, dt=DT)

      delta_x = (x_dot1-x_dot2)/(2*delta)
      B[:,i] = delta_x

    return B

def cost_inter(x, u):
  """intermediate cost function

  Parameters
  ----------
  env: gym.core.Env
    The environment you are try to control. In this homework the 2
    link arm.
  x: np.array
    The state to test. When approximating A you will need to perturb
    this.
  u: np.array
    The command to test. When approximating B you will need to
    perturb this.

  Returns
  -------
  l, l_x, l_xx, l_u, l_uu, l_ux. The first term is the loss, where the remaining terms are derivatives respect to the
  corresponding variables, ex: (1) l_x is the first order derivative d l/d x (2) l_xx is the second order derivative
  d^2 l/d x^2
  """
  
  # cost
  l = np.sum(u**2)
  
  # derivarives of the cost
  action_space = u.shape[0]
  state_space = x.shape[0]

  l_x = np.zeros(state_space)
  l_xx = np.zeros((state_space, state_space))
  l_u = 2 * u
  l_uu = 2 * np.eye(action_space)
  l_ux = np.zeros((action_space, state_space))

  return l, l_x, l_xx, l_u, l_uu, l_ux


def cost_final(env, x):
  """cost function of the last step

  Parameters
  ----------
  env: gym.core.Env
    The environment you are try to control. In this homework the 2
    link arm.
  x: np.array
    The state to test. When approximating A you will need to perturb
    this.

  Returns
  -------
  l, l_x, l_xx The first term is the loss, where the remaining terms are derivatives respect to the
  corresponding variables
  """
  weight = 1e6
  
  err = env.state-env.goal
  l = weight*np.sum(err**2)

  l_x = 2*weight*(err)
  l_xx = 2 * weight * np.eye(4)

  return l, l_x, l_xx


def simulate(env, x0, U):
  tN = U.shape[0]
  state_space = x0.shape[0]
  dt = env.dt

  X = np.zeros((tN, state_space))
  X[0] = copy.deepcopy(x0)
  cost = 0

  for t in range(tN-1):
      X[t+1] = simulate_dynamics_next(env, X[t], U[t])
      l,_,_,_,_,_ = cost_inter(X[t], U[t])
      cost = cost + dt*l

  l_f,_,_ = cost_final(env,X[-1])
  cost = cost + l_f

  return X, cost

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

  action_space = 2 
  state_space = 4
  dt = copy.copy(env.dt)
  regularization_variable = 1.0
  
  # initial guess for the control sequence U
  U = np.zeros((tN, action_space)) 
  
  converged_iterations = True
  x0=env.state.copy()

  list_of_costs = []

  for ii in range(max_iter):

      if converged_iterations == True: 
          X, cost = simulate(sim_env,x0, U)
          oldcost = copy.copy(cost)

          f_x = np.zeros((tN, state_space, state_space)) # df / dx
          f_u = np.zeros((tN, state_space, action_space)) # df / du

          l = np.zeros((tN,1)) # immediate state cost 
          l_x = np.zeros((tN, state_space)) # dl / dx
          l_xx = np.zeros((tN, state_space, state_space)) # d^2 l / dx^2
          l_u = np.zeros((tN, action_space)) # dl / du
          l_uu = np.zeros((tN, action_space, action_space)) # d^2 l / du^2
          l_ux = np.zeros((tN, action_space, state_space)) # d^2 l / du / dx

          for t in range(tN-1):
             
              A = approximate_A(sim_env,X[t], U[t])
              B = approximate_B(sim_env,X[t], U[t])
              f_x[t] = np.eye(state_space) + A * dt
              f_u[t] = B * dt
          
              (l[t], l_x[t], l_xx[t], l_u[t], 
                  l_uu[t], l_ux[t]) = cost_inter(X[t], U[t])
              l[t] *= dt
              l_x[t] *= dt
              l_xx[t] *= dt
              l_u[t] *= dt
              l_uu[t] *= dt
              l_ux[t] *= dt

          l[-1], l_x[-1], l_xx[-1] = cost_final(sim_env,X[-1])

          converged_iterations = False

      V = l[-1].copy() # value function
      V_x = l_x[-1].copy() # dV / dx
      V_xx = l_xx[-1].copy() # d^2 V / dx^2
      k = np.zeros((tN, action_space)) # feedforward modification
      K = np.zeros((tN, action_space, state_space)) # feedback gain

      for t in range(tN-2, -1, -1):

          Q_x = l_x[t] + np.dot(f_x[t].T, V_x) 
          Q_u = l_u[t] + np.dot(f_u[t].T, V_x)

          Q_xx = l_xx[t] + np.dot(f_x[t].T, np.dot(V_xx, f_x[t])) 
          Q_ux = l_ux[t] + np.dot(f_u[t].T, np.dot(V_xx, f_x[t]))
          Q_uu = l_uu[t] + np.dot(f_u[t].T, np.dot(V_xx, f_u[t]))

          # Levenberg-Marquardt heuristic (at end of this loop)
          Q_uu_evals, Q_uu_evecs = np.linalg.eig(Q_uu)
          Q_uu_evals[Q_uu_evals < 0] = 0.0
          Q_uu_evals += regularization_variable
          Q_uu_inv = np.dot(Q_uu_evecs, 
                  np.dot(np.diag(1.0/Q_uu_evals), Q_uu_evecs.T))

          k[t] = -np.dot(Q_uu_inv, Q_u)
          K[t] = -np.dot(Q_uu_inv, Q_ux)
          
          V_x = Q_x - np.dot(K[t].T, np.dot(Q_uu, k[t]))
          V_xx = Q_xx - np.dot(K[t].T, np.dot(Q_uu, K[t]))

      Unew = np.zeros((tN, action_space))
      # calculate the optimal change to the control trajectory
      xnew = x0.copy() # 7a)
      for t in range(tN - 1): 
          # use feedforward (k) and feedback (K) gain matrices 
          # calculated from our value function approximation
          # to take a stab at the optimal control signal
          Unew[t] = U[t] + k[t] + np.dot(K[t], xnew - X[t]) # 7b)
          # given this u, find our next state
          xnew = simulate_dynamics_next(sim_env,xnew, Unew[t]) # 7c)

      # evaluate the new trajectory 
      Xnew, costnew = simulate(sim_env, x0, Unew)

      # Levenberg-Marquardt heuristic, as shown in reference https://raw.githubusercontent.com/studywolf/control/master/studywolf_control/controllers/ilqr.py
      print "iteration = %d"%ii
      print "cost = %.4f"%cost
      print "costnew = %.4f"%costnew

      if costnew < cost: 
          # decrease lambda
          regularization_variable /= LAMB_FACTOR

          X = np.copy(Xnew)
          U = np.copy(Unew)
          oldcost = np.copy(cost)
          cost = np.copy(costnew)
          list_of_costs.append(oldcost)
          converged_iterations = True

          if ii > 0 and ((abs(oldcost-cost)/cost) < EPS_CONVERGE):
              list_of_costs.append(cost)
              break

      else: 
          regularization_variable *= LAMB_FACTOR
          if regularization_variable > LAMB_MAX: 
              break

  return X, U, cost, list_of_costs