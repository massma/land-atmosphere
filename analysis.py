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
    return df[~np.isnan(df.ET)]

def prep_x_data(ds):
    """Take a dataseries DS and make it into the form needed by scikit fit."""
    return ds.to_numpy().reshape(-1, 1)

def reality_diagnostics(_df):
    """Meant to be called on groupby(['year', 'doy'])"""
    if _df.shape[0] < 3:
        return _df.head(n=1)
    else:
        df_out = _df[_df.experiment == 0].copy()
        df_neg = _df[_df.experiment == -1].copy()
        df_pos = _df[_df.experiment == 1].copy()
        sm_neg = df_neg.SM.squeeze()
        sm_0 = df_out.SM.squeeze()
        sm_pos = df_pos.SM.squeeze()
        et_neg = df_neg.ET.squeeze()
        et_0 = df_out.ET.squeeze()
        et_pos = df_pos.ET.squeeze()
        m = LinearRegression()
        m.fit(X=np.array([[sm_neg], [sm_0], [sm_pos]]),
              y=np.array([et_neg, et_0, et_pos]))
        df_out['slope'] = float(m.coef_)
        df_out['neg_difference'] = (et_0 - et_neg) / (sm_0 - sm_neg)
        df_out['pos_difference'] = (et_pos - et_0) / (sm_pos - sm_0)
        df_out['et_neg'] = et_neg
        df_out['sm_neg'] = sm_neg
        df_out['et_pos'] = et_pos
        df_out['sm_pos'] = sm_pos
        return df_out

REALITY = load_experiment('reality-slope')\
          .groupby(['year', 'doy'])\
          .apply(reality_diagnostics)

REALITY = REALITY[~np.isnan(REALITY.slope)]

NSAMPLE = 100

def fit_models(experiments):
    """load experiment and fit model for experiment NAME"""
    for (model_string, d) in experiments.items():
        samples = [d['df'].sample(n=REALITY.shape[0],
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
        if model_string == 'reality':
            d['true-slope'] = d['df'].slope.mean()
            d['true-slopes'] = [x.slope.mean() for x in samples]
    return experiments


experiment_names = ['randomized',
                    'atm',
                    'land']

experiments = dict([(name, {'df' : load_experiment(name)}) for name in experiment_names])

experiments['reality'] = {'df' : REALITY}

experiments = fit_models(experiments)

def rmse(truth, prediction):
    """Return the RMSE between TRUTH (dataseries) and PREDICTION (np array)"""
    return np.sqrt(np.average((prediction-prep_x_data(truth))**2))

def bias(truth, prediction):
    """Return the bias between TRUTH (dataseries and PREDICTION (np array)"""
    return np.average(prediction-prep_x_data(truth))

def model_diagnostics(experiments):
    """Print MODEL diagnostics, labeling them with MODEL_STRING

Mutates each dictionary in EXPERIMENTS, adding slope and slopes."""
    for (model_string, d) in experiments.items():
        d['slope'] = float(d['model'].coef_)
        d['slopes'] = [float(m.coef_) for m in d['models']]
    return experiments

experiments = model_diagnostics(experiments)

def scatter_plot(experiments, experiment='randomized', title=''):
    """return a scatter plot of DATA with regression fits overlaid"""
    facet = sns.relplot(data=experiments[experiment]['df'], x='SM', y='ET')
    ax = facet.ax
    xlim = np.array(ax.get_xlim()).reshape(-1, 1)
    for (name, d) in experiments.items():
        ax.plot(np.squeeze(xlim), np.squeeze(d['model'].predict(xlim)), label=name)
    mean_slope = experiments['reality']['true-slope']
    f = lambda x: experiments[experiment]['model'].intercept_ + mean_slope * x
    ax.plot(np.squeeze(xlim), list(map(f, np.squeeze(xlim))), label='truth')
    plt.title(title)
    plt.legend()
    return


def box_plot(experiments):
    """make a histogram plot"""
    fig, ax = plt.subplots()
    dfs = c.deque()
    for (name, d) in experiments.items():
        _df = pd.DataFrame(d['slopes'], columns=['dET/dSM'])
        _df['name'] = name
        dfs.append(_df)
    _df = pd.DataFrame(d['true-slopes'], columns=['dET/dSM'])
    _df['name'] = 'truth'
    dfs.append(_df)
    df = pd.concat(dfs, ignore_index=True)
    ax = sns.boxplot(x='name', y='dET/dSM', data=df)
    plt.legend()
    ax.set_ylabel('dET/dSM (slope)')
    ax.set_xlabel('Experiment')
    return

box_plot(experiments)

scatter_plot(experiments)
plt.show()
