"""LQR, iLQR and MPC."""

import numpy as np
import scipy.linalg

def simulate_dynamics(env, x, u, dt=1e-5):
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

    xnew, _, _, _, = env._step(u, dt)
    xdot=(xnew-x)/dt

    return np.zeros(x.shape)


def approximate_A(env, x, u, delta=1e-5, dt=1e-5):
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
    # store the original environment
    env0=copy.deepcopy(env)

    # initialize matrix A
    A=np.zeros([4,4])

    # baseline vector at x0 and u0
    x_k0=simulate_dynamics(env, x, u, dt=1e-5)

    env=env0
    x_changed=x
    x_changed[0]+=delta
    x_k1=dt*simulate_dynamics(env, x_changed, u, dt=1e-5)+x_changed
    delta_x=(x_k1-x_k0)/delta
    A[0,:]=delta_x

    env=env0
    x_changed=x
    x_changed[1]+=delta
    x_k1=dt*simulate_dynamics(env, x_changed, u, dt=1e-5)+x_changed
    delta_x=(x_k1-x_k0)/delta
    A[1,:]=delta_x

    env=env0
    x_changed=x
    x_changed[2]+=delta
    x_k1=dt*simulate_dynamics(env, x_changed, u, dt=1e-5)+x_changed
    delta_x=(x_k1-x_k0)/delta
    A[2,:]=delta_x

    env=env0
    x_changed=x
    x_changed[3]+=delta
    x_k1=dt*simulate_dynamics(env, x_changed, u, dt=1e-5)+x_changed
    delta_x=(x_k1-x_k0)/delta
    A[3,:]=delta_x

    return A


def approximate_B(env, x, u, delta=1e-5, dt=1e-5):
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
    env0=copy.deepcopy(env)

    # initialize matrix A
    B=np.zeros([4,1])

    # baseline vector at x0 and u0
    x_k0=simulate_dynamics(env, x, u, dt=1e-5)

    env=env0
    u+=delta
    x_k1=dt*simulate_dynamics(env, x, u, dt=1e-5)+x_changed
    delta_x=(x_k1-x_k0)/delta
    B[:,0]=delta_x

    return B


def calc_lqr_input(env, sim_env):
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
    x=env.state
    u=0.1 # doesn't really matter which value in this case, because dynamics are linear so it doesn't affect estimation of A and B
    sim_env=env
    A=approximate_A(sim_env, x, u, delta=1e-5, dt=1e-5)
    sim_env=env # do I really have to set this, or are the differences negligible?
    B=approximate_B(sim_env, x, u, delta=1e-5, dt=1e-5)
    Q=env.Q
    R=env.R

    # solve ARE equation, discrete time
    P = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))

    #compute the LQR gain
    K = np.matrix(scipy.linalg.inv(B.T.dot(P).dot(B).dot(R)).dot(B.T.dot(P).dot(A)))
    
    u=-K.dot(x)

    return u
