from deeprl_hw3 import imitation
import gym
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
from pprint import pprint

if __name__=='__main__':
	env = gym.make('CartPole-v0')
	env_wrap = imitation.wrap_cartpole(env)

	expert = imitation.load_model('CartPole-v0_config.yaml', 'CartPole-v0_weights.h5f')
	# test_cloned_policy(env, cloned_policy)
	episode_length_list = [1, 10, 50, 100]
	loss_all, accuracy_all = [], []
	mean_reward_all_clones, mean_reward_all_clones_wrap = [], []
	std_reward_all_clones, std_reward_all_clones_wrap = [], []
	mean_reward_expert, std_reward_expert = imitation.test_cloned_policy(env, expert, num_episodes=50, render=False)
	mean_reward_expert_wrap, std_reward_expert_wrap = imitation.test_cloned_policy(env_wrap, expert, num_episodes=50, render=False)

	for curr_num_episodes in episode_length_list
		print "number of episodes {}".format(curr_num_episodes)
		# train on vanilla env
		states_arr, actions_arr = imitation.generate_expert_training_data(expert, env, num_episodes=curr_num_episodes, render= False)
		cloned_policy = Model.from_config(expert.get_config())
		cloned_policy.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
		result_metrics = cloned_policy.fit(states_arr, actions_arr, batch_size=32, epochs=50)
		
		# dump metrics into various lists
		loss_all.append(result_metrics['loss'][-1])
		accuracy_all.append(result_metrics['acc'][-1])

		mean_reward_cloned_curr, std_reward_cloned_curr = imitation.test_cloned_policy(env, cloned_policy, num_episodes=50, render=False)
		mean_reward_all_clones.append(mean_reward_cloned_curr)
		std_reward_cloned_curr.append(std_reward_cloned_curr)

		mean_reward_cloned_curr_wrap, std_reward_cloned_curr_wrap = imitation.test_cloned_policy(env_wrap, cloned_policy, num_episodes=50, render=False)
		mean_reward_all_clones_wrap.append(mean_reward_cloned_curr_wrap)
		std_reward_cloned_curr_wrap.append(std_reward_cloned_curr_wrap)

	pprint(loss_all)
	pprint(accuracy_all)