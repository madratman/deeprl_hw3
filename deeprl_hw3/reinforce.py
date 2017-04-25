import gym

import numpy as np
import tensorflow as tf

from imitation import load_model

def get_total_reward(env, model):
    """compute total reward

    Parameters
    ----------
    env: gym.core.Env
      The environment.
    model: (your action model, which can be anything)

    Returns
    -------
    total_reward: float
    """
    return 0.0


def choose_action(model, observation):
    """choose the action

    Parameters
    ----------
    model: (your action model, which can be anything)
    observation: given observation

    Returns
    -------
    p: float
        probability of action 1
    action: int
        the action you choose
    """
    output = model.predict(np.array([observation]))[0]
    #if output[1] < output[0]:
    p = np.random.uniform()
    if p < output[0]:
      action = 0
    else:
      action = 1

    #print(observation, output[0], output[1], p, action)
    return output[1], action


def reinforce(env, model):
    """Policy gradient algorithm

    Parameters
    ----------
    env: your environment

    Returns
    -------
    total_reward: float
    """
    ALPHA = 0.01 # SGD Learning rate

    opt = tf.train.GradientDescentOptimizer(ALPHA)
    #opt = tf.train.AdamOptimizer(ALPHA)

    action_taken = tf.placeholder(tf.int32, shape=[], name='action_taken')
    relevant_pi = tf.transpose(tf.gather(tf.transpose(model.output), action_taken))
    grads = opt.compute_gradients(tf.log(relevant_pi))

    G_t_var = tf.placeholder(tf.float32, shape=(), name='G_t')
    scaled_grads = [(-G_t_var * grad, var) for grad, var in grads]
    update_op = opt.apply_gradients(scaled_grads)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    state = env.reset()

    # List of state, action, reward tuples.
    episode = []

    rewards = []
    episode_no = 1
    while episode_no < 2500:
      prob, action = choose_action(model, state)
      next_state, reward, done, info = env.step(action)

      episode.append((state, action, reward))
      state = next_state

      if done:
        episode_no += 1
        print(len(episode))
        rewards.append(len(episode))
        # Train
        G_t = 0 # The return
        for state, action, reward in episode[::-1]:
          G_t += reward
          feed_dict = { model.input : np.array([state]), action_taken : action, G_t_var : G_t }
          sess.run(update_op, feed_dict=feed_dict)

        state = env.reset()
        episode = []

        if not episode_no % 100:
          print(sess.run(model.trainable_weights))

    return rewards

if __name__ == "__main__":
  import os

  from matplotlib import pyplot

  env = gym.make("CartPole-v0")

  config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CartPole-v0_config-small.yaml")
  model = load_model(config_path)

  rewards = reinforce(env, model)

  pyplot.plot(rewards)
  pyplot.show()
