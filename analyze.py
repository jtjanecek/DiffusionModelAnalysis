import traceback
import scipy.stats as ss
from scipy.stats import norm
import glob
import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 500
import pandas as pd
import seaborn as sns
from collections import defaultdict
import os
from copy import deepcopy

WORKDIR = '../workdir_converged/'
GROUPS = '../data/groups.csv'
RTS = '../data/rt.csv'
CONDS = '../data/conds.csv'

FILEPATH = os.path.dirname(os.path.abspath(__file__))


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
		
		self._models = sorted([d for d in os.listdir('.') if d not in ['@eaDir', '.DS_Store']])
		self._models = ['model_02']
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
			self.__plot_subj_reps(chains)
		except:
			traceback.print_exc()	
		finally:
			chains.close()

	def __plot_subj_reps(self, chains):
		print(" Plotting subj reps ...")
		old = {}
		young = {}		
		for idx, param in enumerate(['Target', 'Lure (HighSim)', 'Lure (LowSim)', 'Foil']):
			data = np.array(chains.get('deltasubjrep'))
			d = np.squeeze(data[:,:,idx,:])	
			old[param] = d[:,0,:].flatten()
			young[param] = d[:,1,:].flatten()	
			#self.__stats[self.__model][f'delta_rep_{param}_overlap'] = np.sum(np.array(sorted(old[param])) > np.array(sorted(young[param]))) / old[param].shape[0]
			self.__stats[self.__model][f'delta_rep_{param}_overlap'] = np.sum(old[param] < young[param]) / old[param].shape[0]


			#plot_group_post(d, f'deltagroup_{param}')
		old_df = pd.DataFrame(old)
		young_df = pd.DataFrame(young)
		old_df['Group'] = 'Old'
		young_df['Group'] = 'Young'
		df = pd.concat([old_df, young_df])
		df = pd.melt(df, id_vars=['Group'], value_vars=['Target', 'Lure (HighSim)', 'Lure (LowSim)', 'Foil'], var_name = 'Condition', value_name = 'Drift Rate')	

		plt.figure()
		sns.violinplot(x='Condition', y='Drift Rate', hue='Group', data=df)
		plt.title('Subject Representatives')
		plt.legend(loc='upper left')
		plt.savefig('drifts_subj.png')
		plt.close()


	def __plot_posterior_predictives(self, chains):
		print(" Calculating posterior predictives ...")
		print(" Reading ypred ...")
		y = deepcopy(self._rts.values)
		conds = deepcopy(self._conds.values)
		ypred = np.array(chains.get('ypred'))

		n_subj = y.shape[0]
		n_trials = y.shape[1]
		n_conds = len(set(conds.flatten()))

		print(" Plotting ypred acc ...")
		############################### ACC ##############
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
		for i, label in enumerate(['Target','Lure (HighSim)', 'Lure (LowSim)', 'Foil']):
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

		print(" Creating quantiles ...")
		############################### RT ##############
		# subj x condition x neg/pos boundary x 3 quantiles
		# negative_boundary = 0 idx, positive_boundary = 1 idx
		# quantile .1 = 0 idx, quantile .3 = 1 idx, quantile .9 = 2 idx
		y_quant = np.full((n_subj, n_conds, 2, 3), np.nan)		
		ypred_quant = np.full((n_subj, n_conds, 2, 3), np.nan)		

		print('y:',y.shape)
		print('ypred:',ypred.shape)
		# ypred: nchain x subj x trial x samples
		
		for cond_idx in range(n_conds):	
			for subj_idx in range(n_subj):
				# Get the condition labels for this subject condition combo
				s_conds = conds[subj_idx,:]
				print(s_conds.shape)
				# Get the subject y values
				y_subj = y[subj_idx,:].flatten()
				print(y_subj.shape)
				# Get the ypred values for this subj
				ypred_subj = np.squeeze(ypred[:,subj,:,:])
				print(ypred_subj.shape)
			
				# Filter by this specific condition
				y_subj_cond = y_subj[s_conds==cond_idx+1]
				ypred_subj_cond = ypred_subj[:,s_conds==cond_idx+1,:]

				print('y_subj_cond:',y_subj_cond.shape)
				print('ypred_subj_cond:',ypred_subj_cond.shape)

				for neg_pos_idx, boundary_condition in enumerate([y_subj_cond > 0, y_subj_cond < 0]):
					y_boundary = np.abs(y_subj_cond[boundary_condition].flatten())
					ypred_boundary = np.abs(ypred_subj_cond[:,boundary_condition,:].flatten())
					if len(y_boundary) < 10:
						continue
					for quant_idx, q in enumerate([.1, .5, .9]):
						y_quant[subj_idx,cond_idx,neg_pos_idx,quant_idx] = np.quantile(y_boundary,q)
						ypred_quant[subj_idx,cond_idx,neg_pos_idx,quant_idx] = np.quantile(ypred_boundary,q)	
			

	def __plot_group_posteriors(self, chains):
		print(f" Plotting group posteriors for model {self.__model} ...")

		def plot_group_post(posterior_data, label):
			plt.figure()

			group1_data = posterior_data[:,0,:].flatten()
			group2_data = posterior_data[:,1,:].flatten()

			self.__stats[self.__model][f'{label}_old_ci'] = '{:.2f} [{:.2f}, {:.2f}    ]'.format(np.mean(group1_data), *self.__calculate_ci(group1_data))
			self.__stats[self.__model][f'{label}_young_ci'] = '{:.2f} [{:.2f}, {:.2f}    ]'.format(np.mean(group2_data), *self.__calculate_ci(group2_data))

			# Plot posterior 
			plt.hist(group1_data, bins=50, density=True, label='Group 1', alpha=.5)
			plt.hist(group2_data, bins=50, density=True, label='Group 2', alpha=.5)
	
			plt.legend()
			#plt.savefig(f'{label}.png')
			plt.close()

		# Plot main params
		for param in ['alphagroup','betagroup','taugroup']:
			data = np.array(chains.get(param))
			plot_group_post(data, param)

		#### VIOLIN PLOT FOR TAU ONLY
		old = {}
		young = {}		
		data = np.array(chains.get('taugroup'))
		old['Tau'] = data[:,0,:].flatten()
		young['Tau'] = data[:,1,:].flatten()	
		old['Tau1'] = data[:,0,:].flatten()
		young['Tau1'] = data[:,1,:].flatten()	
		old['Tau2'] = data[:,0,:].flatten()
		young['Tau2'] = data[:,1,:].flatten()	
		old['Tau3'] = data[:,0,:].flatten()
		young['Tau3'] = data[:,1,:].flatten()	
				

		old_df = pd.DataFrame(old)
		young_df = pd.DataFrame(young)
		old_df['Group'] = 'Old'
		young_df['Group'] = 'Young'
		df = pd.concat([old_df, young_df])
		df = pd.melt(df, id_vars=['Group'], value_vars=['Tau', 'Tau1', 'Tau2', 'Tau3'], var_name = 'Condition', value_name = 'Seconds')	

		plt.figure()
		sns.violinplot(x='Condition', y='Seconds', hue='Group', data=df)
		plt.xlabel('')
		plt.ylim([0,2])
		plt.xlim([-.5,.5])
		fig = plt.gcf()
		plt.title('Tau Group Means')
		#fig.set_size_inches(6.4/3, 4.8)
		fig.set_size_inches(6.4/4, 4.8)
		ax = plt.gca()
		ax.get_legend().remove()
		plt.savefig('tau_groups.png', bbox_inches='tight')
		plt.close()

		#### VIOLIN PLOT FOR ALL DELTA GROUPS
		old = {}
		young = {}		
		for idx, param in enumerate(['Target', 'Lure (HighSim)', 'Lure (LowSim)', 'Foil']):
			data = np.array(chains.get('deltagroup'))
			d = np.squeeze(data[:,:,idx,:])	
			old[param] = d[:,0,:].flatten()
			young[param] = d[:,1,:].flatten()	
			self.__stats[self.__model][f'delta_{param}_old_ci'] = '{:.2f} [{:.2f}, {:.2f}    ]'.format(np.mean(old[param]), *self.__calculate_ci(old[param]))
			self.__stats[self.__model][f'delta_{param}_young_ci'] = '{:.2f} [{:.2f}, {:.2f}    ]'.format(np.mean(young[param]), *self.__calculate_ci(young[param]))


			
			#plot_group_post(d, f'deltagroup_{param}')
		old_df = pd.DataFrame(old)
		young_df = pd.DataFrame(young)
		old_df['Group'] = 'Old'
		young_df['Group'] = 'Young'
		df = pd.concat([old_df, young_df])
		df = pd.melt(df, id_vars=['Group'], value_vars=['Target', 'Lure (HighSim)', 'Lure (LowSim)', 'Foil'], var_name = 'Condition', value_name = 'Drift Rate')	

		plt.figure()
		sns.violinplot(x='Condition', y='Drift Rate', hue='Group', data=df)
		plt.title('Drift Rate Group Means')
		plt.savefig('drifts.png')
		plt.close()

	def __get_deviance(self, chains):
		data = np.array(chains.get('deviance')).flatten()
		self.__stats[self.__model]['dic'] = np.mean(data)

	def __plot_difference_posteriors(self, chains):
		print(f" Plotting posterior differences for model {self.__model} ...")

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
			#plt.savefig(f'{label}.png')
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

	def save_stats(self):
		os.chdir(FILEPATH)
		df = pd.DataFrame(self.__stats)
		print(f"Saving stats to {os.getcwd()} ...")
		df.to_csv('stats.csv')

	def __calculate_ci(self, datapoints, alpha=.95):
		mean = np.mean(datapoints)
		total_points = datapoints

		data_greater = sorted(datapoints[datapoints > mean])
		data_less = sorted(datapoints[datapoints < mean], reverse=True)
		curr_d = int(len(datapoints) * alpha)
		cur_greater = int(curr_d/2)
		cur_less = int(curr_d/2)
		if int(curr_d/2) > len(data_greater):
			cur_greater = len(data_greater) - 1
			cur_less += int(curr_d/2) - len(data_greater)
		if int(curr_d/2) > len(data_less):
			cur_less = len(data_less) - 1
			cur_greater += int(curr_d/2) - len(data_less)
		return data_less[cur_less], data_greater[cur_greater]

if __name__ == '__main__':
	da = DiffusionAnalyzer(WORKDIR, GROUPS, RTS, CONDS)
	da.plot()	
	
	#da.print_stats()
	da.save_stats()


