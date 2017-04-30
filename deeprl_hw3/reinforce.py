import gym

import numpy as np
import tensorflow as tf

from imitation import load_model

def choose_action(model, observation, sess):
    output = sess.run(model.output, feed_dict={model.input : np.array([observation])})[0]
    return np.random.uniform() > output[0]

def run_episode(env, model, sess):
  state = env.reset()

  # List of state, action, reward tuples.
  episode = []

  while 1:
    action = choose_action(model, state, sess)
    next_state, reward, done, info = env.step(action)

    episode.append((state, action, reward))
    state = next_state

    if done:
      break

  return episode

def reinforce(env, model):
  action_taken = tf.placeholder(tf.int32, shape=(), name='action_taken')
  G_t_var = tf.placeholder(tf.float32, shape=(), name='G_t')

  relevant_pi = tf.gather(tf.transpose(model.output), action_taken)
  update_op = tf.train.AdamOptimizer(ALPHA).minimize(-G_t_var * tf.log(relevant_pi))

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  # Avg, min, max triples (over N training episodes)
  learning_curve = []

  for episode_no in range(1, NUM_EPISODES + 1):
    G_t = 0 # The return
    for state, action, reward in run_episode(env, model, sess)[::-1]:
      G_t += reward
      feed_dict = { model.input : np.array([state]), action_taken : action, G_t_var : G_t }
      sess.run(update_op, feed_dict=feed_dict)

    if not episode_no % EVAL_EVERY:
      print(episode_no, ": evaluating for 100 episodes...", end='')
      rewards = [len(run_episode(env, model, sess)) for i in range(100)]
      learning_curve.append((np.mean(rewards), np.min(rewards), np.max(rewards)))
      print(learning_curve[-1])

  return learning_curve

ALPHA = 2.5e-4 # SGD Learning rate
EVAL_EVERY = 20
NUM_EPISODES = 800

if __name__ == "__main__":
  import os

  from matplotlib import pyplot

  env = gym.make("CartPole-v0")

  config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CartPole-v0_config.yaml")
  model = load_model(config_path)

  avg, mins, maxs = map(np.array, zip(*reinforce(env, model)))

  pyplot.errorbar(EVAL_EVERY * np.array(range(len(avg))), avg, yerr=[avg - mins, maxs - avg], ecolor='red', capsize=3)
  pyplot.xlabel("Training Episode No.")
  pyplot.ylabel("Average Reward (100 Episodes)")
  pyplot.title("Average Reward During Training")
  pyplot.savefig("reinforce_plot.pdf", transparent=True)
  pyplot.show()
