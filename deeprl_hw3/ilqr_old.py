"""LQR, iLQR and MPC."""

import numpy as np
import scipy.linalg
from deeprl_hw3.controllers import DELTA, DT
import copy
from IPython import embed

LAMB_FACTOR=10
EPS_CONVERGE=1e-10
LAMB_MAX=1000

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
  dof = u.shape[0]
  num_states = x.shape[0]

  l = np.sum(u**2)
  # maybe change the cost function later to include how far away we are from the desired state (have to include input with Xgoal in this function)

  # compute derivatives of cost
  l_x = np.zeros(num_states)
  l_xx = np.zeros((num_states, num_states))
  l_u = 2 * u
  l_uu = 2 * np.eye(dof)
  l_ux = np.zeros((dof, num_states))

  # returned in an array for easy multiplication by time step 
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
  num_states = x.shape[0]
  l_x = np.zeros((num_states))
  l_xx = np.zeros((num_states, num_states))

  weight = 1e4 # terminal position cost weight

  err = env.state-env.goal
  l = weight*np.sum(err**2)

  l_x = 2*weight*(err)
  l_xx = 2 * weight * np.eye(4)
  # Final cost only requires these three values
  return l, l_x, l_xx

def simulate(env, x0, U):
  tN = U.shape[0]
  num_states = x0.shape[0]
  dt = env.dt

  X = np.zeros((tN, num_states))
  X[0] = copy.deepcopy(x0)
  cost = 0

  # Run simulation with substeps
  for t in range(tN-1):
      X[t+1] = simulate_dynamics_next(env, X[t], U[t])
      l,_,_,_,_,_ = cost_inter(X[t], U[t])
      cost = cost + dt*l

  # Adjust for final cost, subsample trajectory
  l_f,_,_ = cost_final(env,X[-1])
  cost = cost + l_f
  return X, cost

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

  # tN = U.shape[0] # number of time steps
  dof = 2 # number of degrees of freedom of plant 
  U = np.zeros((tN, dof))
  num_states = dof * 2 # number of states (position and velocity)
  dt = copy.copy(env.dt) # time step

  lamb = 1.0 # regularization parameter
  sim_new_trajectory = True
  x0=env.state.copy()

  list_of_costs = []

  for ii in range(max_iter):

      if sim_new_trajectory == True: 
          # simulate forward using the current control trajectory
          X, cost = simulate(sim_env,x0, U)
          oldcost = copy.copy(cost) # copy for exit condition check

          # now we linearly approximate the dynamics, and quadratically 
          # approximate the cost function so we can use LQR methods 

          # for storing linearized dynamics
          # x(t+1) = f(x(t), u(t))
          f_x = np.zeros((tN, num_states, num_states)) # df / dx
          f_u = np.zeros((tN, num_states, dof)) # df / du
          # for storing quadratized cost function 
          l = np.zeros((tN,1)) # immediate state cost 
          l_x = np.zeros((tN, num_states)) # dl / dx
          l_xx = np.zeros((tN, num_states, num_states)) # d^2 l / dx^2
          l_u = np.zeros((tN, dof)) # dl / du
          l_uu = np.zeros((tN, dof, dof)) # d^2 l / du^2
          l_ux = np.zeros((tN, dof, num_states)) # d^2 l / du / dx
          # for everything except final state
          for t in range(tN-1):
              # x(t+1) = f(x(t), u(t)) = x(t) + dx(t) * dt
              # linearized dx(t) = np.dot(A(t), x(t)) + np.dot(B(t), u(t))
              # f_x = np.eye + A(t)
              # f_u = B(t)
              A = approximate_A(sim_env,X[t], U[t])
              B = approximate_B(sim_env,X[t], U[t])
              f_x[t] = np.eye(num_states) + A * dt
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

          sim_new_trajectory = False

      # optimize things! 
      # initialize Vs with final state cost and set up k, K 
      V = l[-1].copy() # value function
      V_x = l_x[-1].copy() # dV / dx
      V_xx = l_xx[-1].copy() # d^2 V / dx^2
      k = np.zeros((tN, dof)) # feedforward modification
      K = np.zeros((tN, dof, num_states)) # feedback gain

      # NOTE: they use V' to denote the value at the next timestep, 
      # they have this redundant in their notation making it a 
      # function of f(x + dx, u + du) and using the ', but it makes for 
      # convenient shorthand when you drop function dependencies

      # work backwards to solve for V, Q, k, and K
      for t in range(tN-2, -1, -1):

          # NOTE: we're working backwards, so V_x = V_x[t+1] = V'_x

          # 4a) Q_x = l_x + np.dot(f_x^T, V'_x)
          Q_x = l_x[t] + np.dot(f_x[t].T, V_x) 
          # 4b) Q_u = l_u + np.dot(f_u^T, V'_x)
          Q_u = l_u[t] + np.dot(f_u[t].T, V_x)

          # NOTE: last term for Q_xx, Q_uu, and Q_ux is vector / tensor product
          # but also note f_xx = f_uu = f_ux = 0 so they're all 0 anyways.
          
          # 4c) Q_xx = l_xx + np.dot(f_x^T, np.dot(V'_xx, f_x)) + np.einsum(V'_x, f_xx)
          Q_xx = l_xx[t] + np.dot(f_x[t].T, np.dot(V_xx, f_x[t])) 
          # 4d) Q_ux = l_ux + np.dot(f_u^T, np.dot(V'_xx, f_x)) + np.einsum(V'_x, f_ux)
          Q_ux = l_ux[t] + np.dot(f_u[t].T, np.dot(V_xx, f_x[t]))
          # 4e) Q_uu = l_uu + np.dot(f_u^T, np.dot(V'_xx, f_u)) + np.einsum(V'_x, f_uu)
          Q_uu = l_uu[t] + np.dot(f_u[t].T, np.dot(V_xx, f_u[t]))

          # Calculate Q_uu^-1 with regularization term set by 
          # Levenberg-Marquardt heuristic (at end of this loop)
          Q_uu_evals, Q_uu_evecs = np.linalg.eig(Q_uu)
          Q_uu_evals[Q_uu_evals < 0] = 0.0
          Q_uu_evals += lamb
          Q_uu_inv = np.dot(Q_uu_evecs, 
                  np.dot(np.diag(1.0/Q_uu_evals), Q_uu_evecs.T))

          # 5b) k = -np.dot(Q_uu^-1, Q_u)
          k[t] = -np.dot(Q_uu_inv, Q_u)
          # 5b) K = -np.dot(Q_uu^-1, Q_ux)
          K[t] = -np.dot(Q_uu_inv, Q_ux)

          # 6a) DV = -.5 np.dot(k^T, np.dot(Q_uu, k))
          # 6b) V_x = Q_x - np.dot(K^T, np.dot(Q_uu, k))
          V_x = Q_x - np.dot(K[t].T, np.dot(Q_uu, k[t]))
          # 6c) V_xx = Q_xx - np.dot(-K^T, np.dot(Q_uu, K))
          V_xx = Q_xx - np.dot(K[t].T, np.dot(Q_uu, K[t]))

      Unew = np.zeros((tN, dof))
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

      # Levenberg-Marquardt heuristic
      
      print "ii=%d"%ii
      print "cost=%.4f"%cost
      print "costnew=%.4f"%costnew

      if costnew < cost: 
          # decrease lambda (get closer to Newton's method)
          lamb /= LAMB_FACTOR

          X = np.copy(Xnew) # update trajectory 
          U = np.copy(Unew) # update control signal
          oldcost = np.copy(cost)
          cost = np.copy(costnew)
          list_of_costs.append(oldcost)
          sim_new_trajectory = True # do another rollout

          # print("iteration = %d; Cost = %.4f;"%(ii, costnew) + 
          #         " logLambda = %.1f"%np.log(lamb))
          # check to see if update is small enough to exit
          if ii > 0 and ((abs(oldcost-cost)/cost) < EPS_CONVERGE):
              list_of_costs.append(cost)
              print("Converged at iteration = %d; Cost = %.4f;"%(ii,costnew) + 
                      " logLambda = %.1f"%np.log(lamb) + " old cost= %.4f"%(oldcost))
              break

      else: 
          # increase lambda (get closer to gradient descent)
          lamb *= LAMB_FACTOR
          # print("cost: %.4f, increasing lambda to %.4f")%(cost, lamb)
          if lamb > LAMB_MAX: 
              print("lambda > max_lambda at iteration = %d;"%ii + 
                  " Cost = %.4f; logLambda = %.1f"%(cost, 
                                                    np.log(lamb)))
              break

  return X, U, cost, list_of_costs