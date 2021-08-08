import traceback
import scipy.stats as ss
from scipy.stats import norm
import glob
import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 150
import pandas as pd
import seaborn as sns
from collections import defaultdict
import os
from copy import deepcopy
sns.set()


WORKDIR = '../workdir_converged/'
GROUPS = '../data/groups.csv'
RTS = '../data/rt.csv'
CONDS = '../data/conds.csv'


class DiffusionAnalyzer:
	def __init__(self, workdir, groups, rts, conds):

		self._groups = pd.read_csv(groups, index_col=0, float_precision='round_trip')
		self._rts = pd.read_csv(rts, index_col=0, float_precision='round_trip')
		self._conds = pd.read_csv(conds, index_col=0, float_precision='round_trip')

		self._workdir = os.path.abspath(workdir)
		print("Initializing ...")
		if not os.path.isdir(workdir):
			raise Exception("Workdir doesn't exist!")
		os.chdir(workdir)
		
		self._models = sorted(os.listdir('.'))
		if len(self._models) == 0:
			raise Exception('No models found.')
		print(f"Found models: {self._models}")

		self.__stats = defaultdict(dict)

	def plot(self):
		print("Plotting all models ...")
		for model in self._models:
			self._plot_model(model)

	def _plot_model(self, model):
		print(f"Plotting model {model} ...")
		os.chdir(os.path.join(self._workdir,model))
		self.__model = model

		# Read in the model file
		chains = h5py.File(f'{model}.hdf5', 'r')
		
		try:
			if not os.path.isdir('plots'):
				os.makedirs('plots')	
			os.chdir('plots')
			self.__plot_difference_posteriors(chains)
			self.__plot_group_posteriors(chains)
			self.__get_deviance(chains)
			self.__plot_posterior_predictives(chains)
		except:
			traceback.print_exc()	
		finally:
			chains.close()

	def __plot_posterior_predictives(self, chains):
		print("Calculating posterior predictives ...")
		print("Reading ypred ...")
		y = deepcopy(self._rts.values)
		conds = deepcopy(self._conds.values)
		ypred = np.array(chains.get('ypred'))

		n_subj = y.shape[0]
		n_trials = y.shape[1]
		n_conds = len(set(conds.flatten()))

		# Calculate ypred_perc_pos.
		# Each trial will be between 0-1, and will indicate the rate at which 
		# the answer was towards the positive boundary
		ypred_perc_pos = np.zeros((n_subj,n_trials))
		for subj in range(n_subj):
			for trial in range(n_trials):
				ypred_perc_pos[subj,trial] = np.count_nonzero(ypred[:,subj,trial,:].flatten() > 0) / ypred[:,subj,trial,:].flatten().shape[0]

		# Fill in the nans from main y
		ypred_perc_pos[np.isnan(y)] = np.nan

		# We can calculate the accuracy for ypred and for y in each condition
		# n_subj x n_cond x y/ypred
		accs = np.full((n_subj,n_conds,2), np.nan)		
	
		for cond_idx in range(n_conds):
			for subj in range(n_subj):
				# Get the subj-cond acc
				y_subj_cond = y[subj,:][conds[subj,:] == cond_idx+1]
				ypred_subj_cond = ypred_perc_pos[subj,:][conds[subj,:] == cond_idx+1]
				# Remove nans
				y_subj_cond = y_subj_cond[~np.isnan(y_subj_cond)]
				ypred_subj_cond = ypred_subj_cond[~np.isnan(ypred_subj_cond)]

				ypred_subj_cond_acc = np.mean(ypred_subj_cond)
				if cond_idx == 0: # Target, so invert it
					ypred_subj_cond_acc = 1 - ypred_subj_cond_acc
					y_subj_cond_acc = np.sum(y_subj_cond<0) / y_subj_cond.shape[0]
				else:
					y_subj_cond_acc = np.sum(y_subj_cond>0) / y_subj_cond.shape[0]

				accs[subj,cond_idx,0] = y_subj_cond_acc
				accs[subj,cond_idx,1] = ypred_subj_cond_acc
		plt.figure()
		for i, label in enumerate(['Targ','HSim', 'LSim', 'Foil']):
			plt.scatter(accs[:,i,0].flatten(), accs[:,i,1].flatten(), label=label, marker='+')
		plt.plot([0,1], [0,1])
		plt.xlim([-.05,1.05])
		plt.ylim([-.05,1.05])
		plt.title('Posterior Predictive Accuracy')
		plt.legend()
		plt.xlabel("Subject Accuracy")
		plt.ylabel("Modeled Accuracy")
		plt.savefig("posterior_predictive_acc.png")
		plt.close()

		'''
		correct_y = deepcopy(conds)
		correct_y[correct_y == 1] = -1
		correct_y[(correct_y == 2) | (correct_y == 3) | (correct_y == 4)] = 1
		print(conds)
		print(correct_y)
		'''
		

	def __plot_group_posteriors(self, chains):
		print(f"Plotting group posteriors for model {self.__model} ...")

		def plot_group_post(posterior_data, label):
			plt.figure()

			group1_data = posterior_data[:,0,:].flatten()
			group2_data = posterior_data[:,1,:].flatten()

			# Plot posterior 
			plt.hist(group1_data, bins=50, density=True, label='Group 1', alpha=.5)
			plt.hist(group2_data, bins=50, density=True, label='Group 2', alpha=.5)
	
			plt.legend()
			plt.savefig(f'{label}.png')
			plt.close()

		# Plot main params
		for param in ['alphagroup','betagroup','taugroup']:
			data = np.array(chains.get(param))
			plot_group_post(data, param)

		for idx, param in enumerate(['targ', 'hsim', 'lsim', 'foil']):
			data = np.array(chains.get('deltagroup'))
			d = np.squeeze(data[:,:,idx,:])	
			plot_group_post(d, f'deltagroup_{param}')
	

	def __get_deviance(self, chains):
		data = np.array(chains.get('deviance')).flatten()
		self.__stats[self.__model]['dic'] = np.mean(data)

	def __plot_difference_posteriors(self, chains):
		print(f"Plotting posterior differences for model {self.__model} ...")

		def plot_diff(posterior_data, label, prior_mu, prior_sd):
			# Get posterior mu/sd 
			post_mu = np.mean(posterior_data)
			post_sd = np.std(posterior_data)
			
			# Get linspace for plotting for prior
			prior_x = np.linspace(prior_mu-prior_sd*4,prior_mu+prior_sd*4,100)
			prior_y = ss.norm.pdf(prior_x, prior_mu, prior_sd)

			# Calculate BF
			bf = ss.norm.pdf([0], prior_mu, prior_sd)[0] / ss.norm.pdf([0], post_mu, post_sd)[0] 

			plt.figure()

			# Plot posterior 
			plt.hist(posterior_data, bins=50, density=True, label='Posterior')
	
			# Fix xlimits
			cur_xlim = plt.gca().get_xlim()
			min_x = min(cur_xlim[0], prior_mu-prior_sd*1.5)
			max_x = max(cur_xlim[1], prior_mu+prior_sd*1.5)

			# Plot prior
			plt.plot(prior_x, prior_y, label='Prior')

			#plt.xlim([min_x,max_x])
			plt.xlim([cur_xlim[0], cur_xlim[1]])
			plt.legend()
			plt.title(label)
			plt.savefig(f'{label}.png')
			plt.close()

			return bf

		# Plot main params
		for param in [
				{'name':'alphadiff','prior_mu': 0, 'prior_sd': 0.5},
				{'name':'betadiff','prior_mu': 0, 'prior_sd': 0.125},
				{'name':'taudiff','prior_mu': 0, 'prior_sd': 0.15}
				]:
			data = np.array(chains.get(param['name'])).flatten()
			bf = plot_diff(data, param['name'], param['prior_mu'], param['prior_sd'])
			self.__stats[self.__model][f'{param["name"]}_bf'] = bf

		for idx, label in enumerate(['deltadiff_targ', 'deltadiff_hsim', 'deltadiff_lsim', 'deltadiff_foil']):	
			data = np.array(chains.get('deltadiff'))[:,idx,:].flatten()
			prior_mu = 0
			prior_sd = 1.5
			bf = plot_diff(data, label, prior_mu, prior_sd)
			self.__stats[self.__model][f'{label}_bf'] = bf


	def print_stats(self):
		keys = list(self.__stats.values())[0].keys()
	
		for key in keys:
			print()
			print("===========================")
			print(f" --------- {key}")
			for model in self.__stats.keys(): 
				val = self.__stats[model][key]
				print(f'{model}: {val}')

if __name__ == '__main__':
	da = DiffusionAnalyzer(WORKDIR, GROUPS, RTS, CONDS)
	da.plot()	
	
	da.print_stats()


