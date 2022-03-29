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

# causal graph
# fig 1/2: "truth vertical line" and then two histograms of decorrelaed and the naive
# fig 2/3: same, but breaking down each experiment (might need to do mulitple pair wise figures if this gets ugly

data_dir = "./data"

def load_experiment(name):
    """Load experiment NAME."""
    df = pd.read_csv("%s/kelowna-%s.csv" % (data_dir, name))
    return df[~np.isnan(df.ET)]

def prep_x_data(ds):
    """Take a dataseries DS and make it into the form needed by scikit fit."""
    return ds.to_numpy().reshape(-1, 1)

NSAMPLE = 100

def fit_models(experiments):
    """load experiment and fit model for experiment NAME"""
    for (_name, d) in experiments.items():
        samples = [d['df'].sample(n=experiments['reality']['df'].shape[0],
                                  replace=True,
                                  random_state=i)
                   for i in range(NSAMPLE)]
        models = [LinearRegression() for i in range(NSAMPLE)]
        for m,df in zip(models,samples):
            m.fit(X=prep_x_data(df.SM), y=df.ET)
        model = LinearRegression()
        model.fit(X=prep_x_data(d['df'].SM), y=d['df'].ET)
        d['samples'] = samples
        d['models'] = models
        d['model'] = model
    return experiments


experiment_names = ['causal',
                    'randomized',
                    'dynamics',
                    'lai',
                    'temperature',
                    'moisture',
                    'doy',
                    'cc',
                    'doy-lai-temperature',
                    'lai-temperature',
                    'reality']

experiments = dict([(name, {'df' : load_experiment(name)}) for name in experiment_names])
(train, TEST_DF) = train_test_split(experiments['causal']['df'], random_state=0)
experiments['causal'] = {'df' : train}

experiments = fit_models(experiments)


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
        d['prediction'] = d['model'].predict(prep_x_data(TEST_DF.SM))
        d['bias'] = bias(TEST_DF.ET, d['prediction'])
        d['rmse'] = rmse(TEST_DF.ET, d['prediction'])
        d['slope'] = float(d['model'].coef_)
        predictions = [m.predict(prep_x_data(TEST_DF.SM))
                       for m, df in zip(d['models'], d['samples'])]
        d['predictions'] = predictions
        d['slopes'] = [float(m.coef_) for m in d['models']]
        d['biases'] = [bias(TEST_DF.ET, p) for p in predictions]
        d['rmses'] = [rmse(TEST_DF.ET, p) for p in predictions]
    return experiments

experiments = model_diagnostics(experiments)

def scatter_plot(experiments, data=TEST_DF, title=''):
    """return a scatter plot of DATA with regression fits overlaid"""
    facet = sns.relplot(data=data, x='SM', y='ET')
    ax = facet.ax
    xlim = np.array(ax.get_xlim()).reshape(-1, 1)
    for (name, d) in experiments.items():
        ax.plot(np.squeeze(xlim), np.squeeze(d['model'].predict(xlim)), label=name)
    plt.title(title)
    plt.legend()
    return


def hist_plot(experiments, accessor='biases',
              f=lambda x: x, extra_experiment=None):
    """make a histogram plot"""
    fig, ax = plt.subplots()
    ax = sns.histplot(data=f(experiments['randomized'][accessor]), kde=True, label='randomized')

    if extra_experiment:
        ax = sns.histplot(data=f(experiments[extra_experiment][accessor]),
                          kde=True, label='%s confounded' % extra_experiment,
                          color='grey')
    ax = sns.histplot(data=f(experiments['reality'][accessor]), kde=True, ax=ax, label='naive', color='m')
    ylim = ax.get_ylim()
    if accessor == 'biases':
        x = experiments['causal']['bias']
    elif accessor == 'slopes':
        x = experiments['causal']['slope']
    else:
        x = np.nan
    ax.plot(f([x, x]), ylim, label='\"truth\"')
    plt.legend()
    ax.set_xlabel(accessor)
    return

hist_plot(experiments)
hist_plot(experiments, accessor='slopes')
for exp in ['dynamics',
            'lai',
            'temperature',
            'moisture',
            'doy',
            'cc',
            'doy-lai-temperature',
            'lai-temperature']:
    # hist_plot(experiments, f=np.absolute, extra_experiment=exp)
    hist_plot(experiments, accessor='slopes', extra_experiment=exp)

# scatter_plot(experiments)



plt.show()
