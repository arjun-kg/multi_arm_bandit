import numpy as np 
from tqdm import tqdm 
from matplotlib import pyplot as plt 

np.random.seed(4)
no_arms = 10
eps = [0.01,0.05,0.1,0.2,0.3,0.5,0.6,0.7,0.8]
runs = 500

def median_elimination(epsilon, delta):
	q_star = np.random.normal(0,1,no_arms)
	q_dyn_sorted = np.array([np.arange(no_arms),np.zeros(no_arms)])
	timesteps = 0
	eps_l = epsilon/4
	del_l = delta/2

	for i in range((int(np.log2(no_arms)))):
		times = int(4/eps_l**2*np.log(3/del_l))
		cur_avg = []

		for i,arm in enumerate(q_dyn_sorted[0]):
			cur_avg += [np.sum(np.random.normal(q_star[int(arm)],1,size=(times,)))]

		q_dyn_sorted[1,:] = (timesteps*q_dyn_sorted[1,:]+cur_avg)/(times+timesteps)
		q_dyn_sorted = q_dyn_sorted[:,q_dyn_sorted[1,:].argsort()]
		q_dyn_sorted = q_dyn_sorted[:,-int(len(q_dyn_sorted.T)/2):]
		timesteps += times
		eps_l = 3/4*eps_l
		del_l = del_l/2
	return q_dyn_sorted,np.argmax(q_star)

if __name__=="__main__":

	perc_array = []
	for ep in eps:
		perc_opt_arm = 0
		for i in tqdm(range(runs)):

			arm_selected,a_star = median_elimination(epsilon = ep, delta = 0.01)
			if arm_selected[0] == a_star:
				perc_opt_arm += 1
		perc_opt_arm /= runs
		perc_opt_arm *= 100
		perc_array += [perc_opt_arm]

	plt.plot(eps,perc_array)
	plt.xlabel("Epsilon Values")
	plt.ylabel("Percentage Optimal Arm")
	plt.legend("Median Elimination")
	plt.show()



