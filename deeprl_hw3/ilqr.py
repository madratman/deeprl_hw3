"""LQR, iLQR and MPC."""

from deeprl_hw3.controllers import approximate_A, approximate_B
import numpy as np
import scipy.linalg


def simulate_dynamics_next(env, x, u):
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
    xnew, _, _, _, = env._step(u, dt)

    return xnew


def cost_inter(env, x, u):
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
    X[0] = x0
    cost = 0

    # Run simulation with substeps
    for t in range(tN-1):
        X[t+1] = simulate_dynamics_next(env, X[t], U[t])
        l,_,_,_,_,_ = self.cost(X[t], U[t])
        cost = cost + dt * l

    # Adjust for final cost, subsample trajectory
    l_f,_,_ = self.cost_final(X[-1])
    cost = cost + l_f

    return X, cost


def calc_ilqr_input(env, sim_env, tN=50, max_iter=1e6):
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

    self.max_iter = max_iter
    self.lamb_factor = 10
    self.lamb_max = 1000
    self.eps_converge = 0.001 # exit if relative improvement below threshold

    old_target = [None, None]

    return np.zeros((50, 2))

def ilqr(self, x0, U=None): 
  """ use iterative linear quadratic regulation to find a control 
  sequence that minimizes the cost function 

  x0 np.array: the initial state of the system
  U np.array: the initial control trajectory dimensions = [dof, time]
  """
  U = self.U if U is None else U

  tN = U.shape[0] # number of time steps
  dof = self.arm.DOF # number of degrees of freedom of plant 
  num_states = dof * 2 # number of states (position and velocity)
  dt = self.arm.dt # time step

  lamb = 1.0 # regularization parameter
  sim_new_trajectory = True

  for ii in range(self.max_iter):

      if sim_new_trajectory == True: 
          # simulate forward using the current control trajectory
          X, cost = self.simulate(x0, U)
          oldcost = np.copy(cost) # copy for exit condition check

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
              A, B = self.finite_differences(X[t], U[t])
              f_x[t] = np.eye(num_states) + A * dt
              f_u[t] = B * dt
          
              (l[t], l_x[t], l_xx[t], l_u[t], 
                  l_uu[t], l_ux[t]) = self.cost(X[t], U[t])
              l[t] *= dt
              l_x[t] *= dt
              l_xx[t] *= dt
              l_u[t] *= dt
              l_uu[t] *= dt
              l_ux[t] *= dt
          # aaaand for final state
          l[-1], l_x[-1], l_xx[-1] = self.cost_final(X[-1])

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
          _,xnew = self.plant_dynamics(xnew, Unew[t]) # 7c)

      # evaluate the new trajectory 
      Xnew, costnew = self.simulate(x0, Unew)

      # Levenberg-Marquardt heuristic
      if costnew < cost: 
          # decrease lambda (get closer to Newton's method)
          lamb /= self.lamb_factor

          X = np.copy(Xnew) # update trajectory 
          U = np.copy(Unew) # update control signal
          oldcost = np.copy(cost)
          cost = np.copy(costnew)

          sim_new_trajectory = True # do another rollout

          # print("iteration = %d; Cost = %.4f;"%(ii, costnew) + 
          #         " logLambda = %.1f"%np.log(lamb))
          # check to see if update is small enough to exit
          if ii > 0 and ((abs(oldcost-cost)/cost) < self.eps_converge):
              print("Converged at iteration = %d; Cost = %.4f;"%(ii,costnew) + 
                      " logLambda = %.1f"%np.log(lamb))
              break

      else: 
          # increase lambda (get closer to gradient descent)
          lamb *= self.lamb_factor
          # print("cost: %.4f, increasing lambda to %.4f")%(cost, lamb)
          if lamb > self.lamb_max: 
              print("lambda > max_lambda at iteration = %d;"%ii + 
                  " Cost = %.4f; logLambda = %.1f"%(cost, 
                                                    np.log(lamb)))
              break

  return X, U, cost