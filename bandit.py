import numpy as np 
from tqdm import tqdm 
from matplotlib import pyplot as plt 
import pdb

TS = 1000
RUNS = 2000
np.random.seed(4)
NO_ARMS = 10

class bandit_problem: 
	def __init__(self):
		self.q_star = np.random.normal(0,1,NO_ARMS)
		self.no_arms = len(self.q_star)
		self.q_hat = np.zeros(NO_ARMS)
		self.pulls_counter = np.ones(NO_ARMS)
		self.reward_array = []
		self.action_array = []
		self.epsilon = 0
		self.T = 1

	def sample(self,arm):
		#Arms 0 to 9
		r = np.random.normal(self.q_star[arm],1)
		self.reward_array += [r]
		self.action_array += [arm]
		return r

	def pull_arms(self,times=1,arms=None):
		try:
			arms_est = np.zeros(len(arms))
		except: 
			arms = np.arange(self.no_arms)
			arms_est = np.zeros(len(arms))
		
		for i,arm in enumerate(arms):
			for j in range(times):
				arms_est[i] += self.sample(int(arm))
		return arms_est

	def epsilon_greedy(self):
		if self.epsilon > np.random.rand(1):
			action = np.random.randint(10)
		else:
			action = np.argmax(self.q_hat)
		return action

	def softmax(self):
		softmax_array = np.exp(self.q_hat/self.T)/np.sum(np.exp(self.q_hat/self.T))
		action = np.random.choice(NO_ARMS,p=softmax_array)
		return action

	def ucb(self,timesteps):
		ucb_array = self.q_hat + 2*np.sqrt(np.log(timesteps)/self.pulls_counter)
		action = np.argmax(ucb_array)
		return action 

	def median_elimination(self, epsilon, delta):
		q_dyn_sorted = np.array([np.arange(self.no_arms),np.zeros(self.no_arms)])
		timesteps = 0
		eps_l = epsilon/4
		del_l = delta/2

		for i in range((int(np.log2(self.no_arms)))):
			times = int(4/eps_l**2*np.log(3/del_l))
			# pdb.set_trace()
			cur_avg = self.pull_arms(times=times, arms= q_dyn_sorted[0])
			q_dyn_sorted[1,:] = (timesteps*q_dyn_sorted[1,:]+cur_avg)/(times+timesteps)
			q_dyn_sorted = q_dyn_sorted[:,q_dyn_sorted[1,:].argsort()]
			q_dyn_sorted = q_dyn_sorted[:,-int(len(q_dyn_sorted.T)/2):]
			timesteps += times
			eps_l = 3/4*eps_l
			del_l = del_l/2
		return q_dyn_sorted

	def run(self, timesteps, algo = 'eg'):
		if algo == 'ucb':
			self.q_hat = self.pull_arms()
			n_pulls = self.no_arms

		for i in range(timesteps):

			if algo == 'eg':
				action = self.epsilon_greedy()
			elif algo == 'sm':
				action = self.softmax()
			elif algo == 'ucb':
				action = self.ucb(n_pulls)
				n_pulls += 1
				if n_pulls == timesteps+1: break

			reward = self.sample(action)
			n = self.pulls_counter[action]
			self.q_hat[action] = (n*self.q_hat[action]+ reward)/(n+1)
			self.pulls_counter[action] += 1

		return 

if __name__=="__main__":
	
	'''
	Epsilon-Greedy
	'''

	eps = [0,0.1,0.01]	

	for epsilon in eps:

		avg_reward = np.zeros(TS)
		perc_opt_arm  = np.zeros(TS)

		for i in tqdm(range(RUNS)):
			agent = bandit_problem()
			a_star = np.argmax(agent.q_star)
			agent.epsilon = epsilon
			agent.run(TS)
			avg_reward += agent.reward_array
			perc_opt_arm[agent.action_array == a_star] += 1

		avg_reward /= RUNS
		perc_opt_arm /= RUNS
		perc_opt_arm *= 100


		plt.figure(1)
		plt.plot(avg_reward)
		plt.xlabel("steps")
		plt.ylabel("Average Reward")

		plt.ylim(ymin = 0,ymax = 1.5)

		plt.figure(2)
		plt.plot(perc_opt_arm)
		plt.xlabel("steps")
		plt.ylabel("% Optimal Arm")
		plt.ylim(ymin = 0,ymax = 100)
	
	plt.figure(1)
	plt.legend(["Epsilon = {}".format(x) for x in eps])
	plt.figure(2)
	plt.legend(["Epsilon = {}".format(x) for x in eps])
	plt.show(1)
	plt.show(2)

	'''
	Softmax
	'''

	temps = [1e-1,1, 10, 10000]	

	for temp in temps:

		avg_reward = np.zeros(TS)
		perc_opt_arm  = np.zeros(TS)

		for i in tqdm(range(RUNS)):
			agent = bandit_problem()
			a_star = np.argmax(agent.q_star)
			agent.T = temp
			agent.run(algo="sm",timesteps = TS)
			avg_reward += agent.reward_array
			perc_opt_arm[agent.action_array == a_star] += 1

		avg_reward /= RUNS
		perc_opt_arm /= RUNS
		perc_opt_arm *= 100


		plt.figure(1)
		plt.plot(avg_reward)
		plt.xlabel("steps")
		plt.ylabel("Average Reward")

		plt.ylim(ymin = 0,ymax = 1.5)

		plt.figure(2)
		plt.plot(perc_opt_arm)
		plt.xlabel("steps")
		plt.ylabel("% Optimal Arm")
		plt.ylim(ymin = 0,ymax = 100)
	
	plt.figure(1)
	plt.legend(["Temperature = {}".format(x) for x in temps])
	plt.figure(2)
	plt.legend(["Temperature = {}".format(x) for x in temps])
	plt.show(1)
	plt.show(2)

	'''
	UCB Compared with Softmax and Epsilon Greedy
	'''
	avg_reward = np.zeros(TS)
	perc_opt_arm  = np.zeros(TS)

	for i in tqdm(range(RUNS)):
		agent = bandit_problem()
		a_star = np.argmax(agent.q_star)
		agent.run(algo = "ucb", timesteps = TS)
		avg_reward += agent.reward_array
		perc_opt_arm[agent.action_array == a_star] += 1

	avg_reward /= RUNS
	perc_opt_arm /= RUNS
	perc_opt_arm *= 100

	plt.figure(1)
	plt.plot(avg_reward)
	plt.figure(2)
	plt.plot(perc_opt_arm)

	for i in tqdm(range(RUNS)):
		agent = bandit_problem()
		a_star = np.argmax(agent.q_star)
		agent.T = 0.1
		agent.run(algo = "sm", timesteps = TS)
		avg_reward += agent.reward_array
		perc_opt_arm[agent.action_array == a_star] += 1

	avg_reward /= RUNS
	perc_opt_arm /= RUNS
	perc_opt_arm *= 100

	plt.figure(1)
	plt.plot(avg_reward)
	plt.figure(2)
	plt.plot(perc_opt_arm)

	for i in tqdm(range(RUNS)):
		agent = bandit_problem()
		a_star = np.argmax(agent.q_star)
		agent.epsilon = 0.1
		agent.run(algo = "eg", timesteps = TS)
		avg_reward += agent.reward_array
		perc_opt_arm[agent.action_array == a_star] += 1

	avg_reward /= RUNS
	perc_opt_arm /= RUNS
	perc_opt_arm *= 100

	plt.figure(1)
	plt.plot(avg_reward)
	plt.legend(["UCB1", "Softmax (T = 0.1)", "Epsilon-Greedy (eps = 0.1)"])
	plt.xlabel("steps")
	plt.ylabel("Average Reward")
	plt.ylim(ymin = -1,ymax = 6)
	plt.figure(2)
	plt.plot(perc_opt_arm)
	plt.legend(["UCB1", "Softmax (T = 0.1)", "Epsilon-Greedy (eps = 0.1)"])
	plt.xlabel("steps")
	plt.ylabel("% Optimal Arm")
	plt.ylim(ymin = 0,ymax = 100)

	plt.show()

	'''
	Median Elimination
	'''
	agent = bandit_problem()
	a_star = np.argmax(agent.q_star)
	_ = agent.median_elimination(epsilon = 0.5, delta = 0.01)
	avg_reward = np.array(agent.reward_array)
	perc_opt_arm = np.zeros_like(avg_reward)
	perc_opt_arm[agent.action_array == a_star] = 1

	for i in tqdm(range(RUNS-1)):
		agent = bandit_problem()
		a_star = np.argmax(agent.q_star)
		_ = agent.median_elimination(epsilon = 0.5, delta = 0.01)
		avg_reward += agent.reward_array
		perc_opt_arm[agent.action_array == a_star] += 1

	avg_reward /= RUNS
	perc_opt_arm /= RUNS
	perc_opt_arm *= 100


	plt.figure(1)
	plt.plot(avg_reward)
	plt.xlabel("steps")
	plt.ylabel("Average Reward")
	plt.legend(["Median Elimination"])

	plt.ylim(ymin = 0,ymax = 2.0)

	plt.figure(2)
	plt.plot(perc_opt_arm)
	plt.xlabel("steps")
	plt.ylabel("% Optimal Arm")
	plt.ylim(ymin = 0,ymax = 100)
	plt.legend(["Median Elimination"])
	plt.show()
