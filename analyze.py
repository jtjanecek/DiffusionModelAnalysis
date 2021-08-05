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
sns.set()


WORKDIR = '../workdir_converged/'


class DiffusionAnalyzer:
	def __init__(self, workdir):
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
		except:
			traceback.print_exc()	
		finally:
			chains.close()

	def __plot_group_posteriors(self, chains):
		print(f"Plotting group posteriors for model {self.__model} ...")

		def plot_group_post(posterior_data, label):
			plt.figure()
			print(posterior_data.shape)

			group1_data = posterior_data[:,0,:].flatten()
			group2_data = posterior_data[:,1,:].flatten()
			print(group1_data.shape)
			print(group2_data.shape)

			# Plot posterior 
			plt.hist(group1_data, bins=50, density=True, label='Group 1', alpha=.5)
			plt.hist(group2_data, bins=50, density=True, label='Group 2', alpha=.5)
	
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
	da = DiffusionAnalyzer(WORKDIR)
	da.plot()	
	
	da.print_stats()


