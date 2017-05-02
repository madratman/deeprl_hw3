"""Functions for imitation learning."""
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

from keras.models import model_from_yaml
import numpy as np
import time

def load_model(model_config_path, model_weights_path=None):
    """Load a saved model.

    Parameters
    ----------
    model_config_path: str
      The path to the model configuration yaml file. We have provided
      you this file for problems 2 and 3.
    model_weights_path: str, optional
      If specified, will load keras weights from hdf5 file.

    Returns
    -------
    keras.models.Model
    """
    with open(model_config_path, 'r') as f:
        model = model_from_yaml(f.read())

    if model_weights_path is not None:
        model.load_weights(model_weights_path)

    model.summary()

    return model


# def generate_expert_training_data(expert, env, num_episodes=100, render=True):
#     """Generate training dataset.

#     Parameters
#     ----------
#     expert: keras.models.Model
#       Model with expert weights.
#     env: gym.core.Env
#       The gym environment associated with this expert.
#     num_episodes: int, optional
#       How many expert episodes should be run.
#     render: bool, optional
#       If present, render the environment, and put a slight pause after
#       each action.

#     Returns
#     -------
#     expert_dataset: ndarray(states), ndarray(actions)
#       Returns two lists. The first contains all of the states. The
#       second contains a one-hot encoding of all of the actions chosen
#       by the expert for those states.
#     """

#     states_arr = np.empty((0, 4))
#     actions_arr = np.empty((0, 2))

#     for i in range(num_episodes):
#         print('generate_expert_training_data :: episode {}'.format(i))
#         state = env.reset()
#         state = np.reshape(state, (1, state.shape[0]))
#         if render:
#             env.render()
#             time.sleep(.1)
#         is_done = False
#         while not is_done:
#             # print(state.shape)
#             # print(state.shape[0])
#             # print(np.reshape(state, (1, state.shape[0])).shape)

#             # action = np.argmax(expert.predict_on_batch(state[np.newaxis, ...])[0])
#             action = np.argmax(expert.predict(state, batch_size=1)[0,:])
#             # print("action", action)
#             if action == 0:
#                 action_one_hot = np.array([[0., 1.]])
#             else:
#                 action_one_hot = np.array([[1., 0.]])
#             next_state, reward, is_done, _ = env.step(action)
#             next_state = np.reshape(next_state, (1, next_state.shape[0]))
#             states_arr = np.append(states_arr, state, axis=0)
#             actions_arr = np.append(actions_arr, action_one_hot, axis=0)
#             # print(state.shape, states_arr.shape)
#             # print(action_one_hot.shape, actions_arr.shape)
#             state = next_state
#             if render:
#                 env.render()
#                 time.sleep(.1)

#     print(' \n DONE generate_expert_training_data \n')
#     return states_arr, actions_arr
def generate_expert_training_data(expert, env, num_episodes=100, render=True):
    """Generate training dataset.

    Parameters
    ----------
    expert: keras.models.Model
      Model with expert weights.
    env: gym.core.Env
      The gym environment associated with this expert.
    num_episodes: int, optional
      How many expert episodes should be run.
    render: bool, optional
      If present, render the environment, and put a slight pause after
      each action.

    Returns
    -------
    expert_dataset: ndarray(states), ndarray(actions)
      Returns two lists. The first contains all of the states. The
      second contains a one-hot encoding of all of the actions chosen
      by the expert for those states.
    """

    states_arr = np.empty((0,4))
    actions_arr = np.empty((0, 2))

    for i in range(num_episodes):
        currState = env.reset()
        if render:
            env.render()
        currState = np.reshape(currState, (1,currState.size))
        is_terminal = False
        while is_terminal == False:
            actionP = expert.predict(currState, batch_size = 1)
            action = np.argmax(actionP[0,:])
            nextState, reward, is_terminal, _ = env.step(action)
            nextState = np.reshape(nextState, (1,nextState.size))
            if render:
                env.render()       
            action_oh = np.zeros((1,2))
            np.put(action_oh,action,1)        
            states = np.append(states, currState, axis = 0)
            actions = np.append(actions,action_oh, axis = 0)
            
            currState = nextState 
        # time.sleep(0.1)

    return states, actions

def test_cloned_policy(env, cloned_policy, num_episodes=50, render=True):
    """Run cloned policy and collect statistics on performance.

    Will print the rewards for each episode and the mean/std of all
    the episode rewards.

    Parameters
    ----------
    env: gym.core.Env
      The CartPole-v0 instance.
    cloned_policy: keras.models.Model
      The model to run on the environment.
    num_episodes: int, optional
      Number of test episodes to average over.
    render: bool, optional
      If true, render the test episodes. This will add a small delay
      after each action.
    """
    total_rewards = []

    for i in range(num_episodes):
        print('Starting episode {}'.format(i))
        total_reward = 0
        state = env.reset()
        if render:
            env.render()
            time.sleep(.1)
        is_done = False
        while not is_done:
            action = np.argmax(
                cloned_policy.predict_on_batch(state[np.newaxis, ...])[0])
            state, reward, is_done, _ = env.step(action)
            total_reward += reward
            if render:
                env.render()
                time.sleep(.1)
        print(
            'Total reward: {}'.format(total_reward))
        total_rewards.append(total_reward)

    print('Average total reward: {} (std: {})'.format(np.mean(total_rewards), np.std(total_rewards)))

    return np.mean(total_rewards), np.std(total_rewards)


def wrap_cartpole(env):
    """Start CartPole-v0 in a hard to recover state.

    The basic CartPole-v0 starts in easy to recover states. This means
    that the cloned model actually can execute perfectly. To see that
    the expert policy is actually better than the cloned policy, this
    function returns a modified CartPole-v0 environment. The
    environment will start closer to a failure state.

    You should see that the expert policy performs better on average
    (and with less variance) than the cloned model.

    Parameters
    ----------
    env: gym.core.Env
      The environment to modify.

    Returns
    -------
    gym.core.Env
    """
    unwrapped_env = env.unwrapped
    unwrapped_env.orig_reset = unwrapped_env._reset

    def harder_reset():
        unwrapped_env.orig_reset()
        unwrapped_env.state[0] = np.random.choice([-1.5, 1.5])
        unwrapped_env.state[1] = np.random.choice([-2., 2.])
        unwrapped_env.state[2] = np.random.choice([-.17, .17])
        return unwrapped_env.state.copy()

    unwrapped_env._reset = harder_reset

    return env

