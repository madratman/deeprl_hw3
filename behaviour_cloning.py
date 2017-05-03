from deeprl_hw3 import imitation
import gym
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
from pprint import pprint
import os

def dump_list_to_file(some_list, filename):
	script_dir = os.path.dirname(os.path.realpath(__file__))
	logdir = os.path.join(script_dir, 'results_behaviour_cloning')
	if not os.path.exists(logdir):
		os.makedirs(logdir)

	f = open(os.path.join(logdir, filename), 'w')
	for each_thing in some_list:
		f.write("%s\n" % each_thing)

if __name__=='__main__':
	# fancy printing 
	RED = '\033[91m'
	BOLD = '\033[1m'
	ENDC = '\033[0m'        
	LINE = "%s%s##############################################################################%s" % (RED, BOLD, ENDC)

	env = gym.make('CartPole-v0')
	env_wrap = gym.make('CartPole-v0')
	env_wrap = imitation.wrap_cartpole(env_wrap)

	expert = imitation.load_model('CartPole-v0_config.yaml', 'CartPole-v0_weights.h5f')
	# test_cloned_policy(env, cloned_policy)
	episode_length_list = [1, 10, 50, 100]
	loss_all, accuracy_all = [], []
	mean_reward_clones_list, mean_reward_clones_wrap_list = [], []
	std_reward_clones_list, std_reward_clones_wrap_list = [], []

	for curr_num_episodes in episode_length_list:
		str_1 = "Imitator with number of episodes = {}".format(curr_num_episodes)
		msg = "\n%s\n" % (LINE) + "%s%s\n" % (BOLD, str_1) + "%s\n" % (LINE)
		print(str(msg))

		# train on vanilla env
		states_arr, actions_arr = imitation.generate_expert_training_data(expert, env, num_episodes=curr_num_episodes, render= False)
		cloned_policy = Model.from_config(expert.get_config())
		cloned_policy.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
		# print states_arr.shape, actions_arr.shape
		result_metrics = cloned_policy.fit(states_arr, actions_arr, batch_size=32, epochs=50)
		
		# dump metrics into various lists
		loss_all.append(result_metrics.history['loss'][-1])
		accuracy_all.append(result_metrics.history['acc'][-1])

		mean_reward_cloned_curr, std_reward_cloned_curr = imitation.test_cloned_policy(env, cloned_policy, num_episodes=50, render=False)
		mean_reward_clones_list.append(mean_reward_cloned_curr)
		std_reward_clones_list.append(std_reward_cloned_curr)

		mean_reward_cloned_curr_wrap, std_reward_cloned_curr_wrap = imitation.test_cloned_policy(env_wrap, cloned_policy, num_episodes=50, render=False)
		mean_reward_clones_wrap_list.append(mean_reward_cloned_curr_wrap)
		std_reward_clones_wrap_list.append(std_reward_cloned_curr_wrap)

	# test expert
	mean_reward_expert, std_reward_expert = imitation.test_cloned_policy(env, expert, num_episodes=50, render=False)
	mean_reward_expert_wrap, std_reward_expert_wrap = imitation.test_cloned_policy(env_wrap, expert, num_episodes=50, render=False)

	print "\n\nExpert stats"
	print("mean_reward_expert {} , std_reward_expert {}".format(mean_reward_expert, std_reward_expert))
	print("mean_reward_expert_wrap {} , std_reward_expert_wrap {}".format(mean_reward_expert_wrap, std_reward_expert_wrap))

	dump_list_to_file(loss_all, 'loss_all.txt')
	dump_list_to_file(accuracy_all, 'accuracy_all.txt')
	dump_list_to_file(mean_reward_clones_list, 'mean_reward_clones_list.txt')
	dump_list_to_file(mean_reward_clones_wrap_list, 'mean_reward_clones_wrap_list.txt')
	dump_list_to_file(std_reward_clones_list, 'std_reward_clones_list.txt')
	dump_list_to_file(std_reward_clones_wrap_list, 'std_reward_clones_wrap_list.txt')
