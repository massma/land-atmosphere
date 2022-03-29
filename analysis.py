import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import collections as c
import seaborn as sns
import random
sns.set_theme()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data_dir = "./data"

def load_experiment(name):
    """Load experiment NAME."""
    df = pd.read_csv("%s/kelowna-%s.csv" % (data_dir, name))
    return {'df' : df[~np.isnan(df.ET)]}

def prep_x_data(ds):
    """Take a dataseries DS and make it into the form needed by scikit fit."""
    return ds.to_numpy().reshape(-1, 1)

def fit_models(experiments):
    """load experiment and fit model for experiment NAME"""
    for (_name, d) in experiments.items():
        model = LinearRegression()
        model.fit(X=prep_x_data(d['df'].SM), y=d['df'].ET)
        d['model'] = model
    return experiments

NSAMPLE = 10

experiment_names = ['causal',
                    'randomized',
                    'dynamics',
                    'lai',
                    'temperature',
                    'moisture',
                    'doy',
                    'cc',
                    'reality']

experiments = dict([(name, load_experiment(name)) for name in experiment_names])
samples = dict([('sample%04d' % i,
                 {'df' :
                  experiments['randomized']['df']\
                  .sample(n=experiments['reality']['df'].shape[0],
                          random_state=i)})
                 for i in range(NSAMPLE)])
samples['causal'] = experiments['causal']

experiments = fit_models(experiments)
samples = fit_models(samples)


def rmse(truth, prediction):
    """Return the RMSE between TRUTH (dataseries) and PREDICTION (np array)"""
    return np.sqrt(np.average((prediction-prep_x_data(truth))**2))

def bias(truth, prediction):
    """Return the bias between TRUTH (dataseries and PREDICTION (np array)"""
    return np.average(prediction-prep_x_data(truth))

def model_diagnostics(experiments):
    """Print MODEL diagnostics, labeling them with MODEL_STRING

Mutates each dictionary in EXPERIMENTS, adding prediction, bias, and rmse"""
    causal = experiments['causal']['df']
    for (model_string, d) in experiments.items():
        d['prediction'] = d['model'].predict(prep_x_data(causal.SM))
        d['bias'] = bias(causal.ET, d['prediction'])
        d['rmse'] = rmse(causal.ET, d['prediction'])
        print("%s bias: %f" % (model_string, d['bias']))
        print("%s rmse: %f" % (model_string, d['rmse']))
        ## below plot not very useful
        # facet = sns.displot(x=causal.ET, y=np.squeeze(d['prediction']))
        # ax = facet.ax
        # ax.set_title('%s' % model_string)
        # xlim = ax.get_xlim()
        # ylim = ax.get_ylim()
        # lim = [max([xlim[0], ylim[0]]), min([xlim[1], ylim[1]])]
        # ax.plot(lim, lim)
    return experiments

experiments = model_diagnostics(experiments)
samples = model_diagnostics(samples)

samples.pop('causal')

mean_sampling_bias = np.average([d['bias'] for d in samples.values()])
std_sampling_bias = np.std([d['bias'] for d in samples.values()])


def scatter_plot(experiments, samples=False, scatter='causal'):
    """return a scatter plot of DATA with regression fits overlaid"""
    facet = sns.relplot(data=experiments[scatter]['df'], x='SM', y='ET')
    ax = facet.ax
    xlim = np.array(ax.get_xlim()).reshape(-1, 1)
    if samples:
        for d in samples.values():
            ax.plot(np.squeeze(xlim), np.squeeze(d['model'].predict(xlim)), label=None,linewidth=0.4)
    for (name, d) in experiments.items():
        ax.plot(np.squeeze(xlim), np.squeeze(d['model'].predict(xlim)), label=name)
    plt.title('scattered data: %s' % scatter)
    plt.legend()
    return

scatter_plot(experiments, scatter='causal')
scatter_plot(experiments, scatter='reality')
plt.show()
def sum_biases(keys, experiments):
    """add up all the biases in EXPERIMENTS corresponding to keys"""
    sum = 0.0
    for x in keys:
        sum = sum + experiments[x]['bias']
    return sum
exps = set(experiment_names) - {'reality', 'causal'}
biases = sum_biases(exps, experiments) + mean_sampling_bias
plt.show()
