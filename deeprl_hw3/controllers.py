"""LQR, iLQR and MPC."""

import numpy as np
import scipy.linalg

DELTA = 1e-5
DT = 1e-3

# Discrete vs continuous controller
DISCRETE = True
# If True, will augment state to deal with affine term
AUGMENT = False
# If True, will directly use affine LQR (only for discrete)
AFFINE = True

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
    env.state = x.copy()
    next_state, _, _, _ = env._step(u)
    if DISCRETE:
      return next_state

    return (next_state - x) / dt

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
    A = np.zeros((len(x), len(x)))
    for i in range(len(x)):
      x_test = x.copy()
      x1 = simulate_dynamics(env, x_test, u)
      x_test[i] += delta
      x2 = simulate_dynamics(env, x_test, u)
      deriv = (x2 - x1) / delta
      A[:, i] = deriv
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
    B = np.zeros((len(x), len(u)))
    for i in range(len(u)):
      u_test = u.copy()
      x1 = simulate_dynamics(env, x, u_test)
      u_test[i] += delta
      x2 = simulate_dynamics(env, x, u_test)
      deriv = (x2 - x1) / delta
      B[:, i] = deriv
    return B

def solve_riccati(A, B, c, Q, R):
  P1 = np.zeros(np.shape(A))
  P2 = np.zeros(len(A))

  while 1:
    inv = np.linalg.inv(R + B.T.dot(P1.dot(B)))
    K = -inv.dot(B.T).dot(P1.dot(A))
    k = -inv.dot(0.5 * P2.dot(B) + B.T.dot(P1).dot(c))
    new_P1 = Q + K.T.dot(R).dot(K) + (A + B.dot(K)).T.dot(P1).dot(A + B.dot(K))
    new_P2 = 2*k.T.dot(R).dot(K) + 2 * (B.dot(k) + c).T.dot(P1).dot(A + B.dot(K)) + P2.dot(A) + P2.dot(B).dot(K)

    if np.linalg.norm(P1 - new_P1) < 1e-6 and \
       np.linalg.norm(P2 - new_P2) < 1e-6:
      break

    P1 = new_P1
    P2 = new_P2

  return P1, P2

def calc_lqr_input(env, sim_env, prev_u):
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
    # The direct affine solution is only implemented for discrete LQR.
    assert not AFFINE or DISCRETE
    # Cannot both augment and do the direct affine solution.
    assert not AFFINE or not AUGMENT

    affine = simulate_dynamics(sim_env, env.state, prev_u)
    A = approximate_A(sim_env, env.state, prev_u)
    B = approximate_B(sim_env, env.state, prev_u)

    if DISCRETE:
      affine = affine - A.dot(env.state) - B.dot(prev_u) - env.goal + A.dot(env.goal)
    else:
      # This causes the CARE to be ill conditioned?! I think it is right though...
      #affine = affine - A.dot(env.state) - B.dot(prev_u) - env.goal + A.dot(env.goal)
      pass

    if AUGMENT:
      new_state = np.hstack((env.state, np.array((1,))))
      new_A = np.hstack((A, np.array([affine]).T))
      new_A = np.vstack((new_A, np.array([np.zeros(len(env.state) + 1)])))
      new_B = np.vstack((B, np.zeros(len(prev_u))))
      new_Q = np.zeros((len(env.state) + 1, len(env.state) + 1))
      new_Q[:len(env.state), :len(env.state)] = env.Q

      # HAX
      if DISCRETE:
        new_A[len(env.state), len(env.state)] = 0.92
      else:
        new_A[len(env.state), len(env.state)] = -20

    if DISCRETE:
      solver = scipy.linalg.solve_discrete_are
    else:
      solver = scipy.linalg.solve_continuous_are

    Q = env.Q
    if AUGMENT:
      A = new_A
      B = new_B
      Q = new_Q

    error = env.state - env.goal
    if AUGMENT:
      error = np.hstack((error, np.array((1.,))))

    if AFFINE:
      P1, P2 = solve_riccati(A, B, affine, Q, env.R)
      return -np.linalg.solve(env.R + B.T.dot(P1).dot(B), 0.5 * P2.dot(B) + B.T.dot(P1).dot(affine) + B.T.dot(P1).dot(A).dot(error))

    else:
      P = solver(A, B, Q, env.R)

      if DISCRETE:
        lhs = env.R + B.T.dot(P).dot(B)
        rhs = B.T.dot(P).dot(A)
        rhsa = B.T.dot(P)
      else:
        lhs = env.R
        rhs = B.T.dot(P)

      return -np.linalg.solve(lhs, rhs.dot(error))
