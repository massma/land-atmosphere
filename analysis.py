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

reality = load_experiment('reality-slope')\
          .groupby(['year', 'doy'])\
          .apply(reality_diagnostics)

reality = reality[~np.isnan(reality.slope)]

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
                    'atm',
                    'land',
                    'reality']

experiments = dict([(name, {'df' : load_experiment(name)}) for name in experiment_names])
(TRAIN, TEST_DF) = train_test_split(experiments['causal']['df'], random_state=0)
experiments['causal'] = {'df' : TRAIN}

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
    ax = sns.histplot(data=f(experiments['causal'][accessor]),
                      kde=True, label='causal')

    if extra_experiment:
        ax = sns.histplot(data=f(experiments[extra_experiment][accessor]),
                          kde=True, label='%s confounded' % extra_experiment,
                          color='grey')
    ax = sns.histplot(data=f(experiments['reality'][accessor]), kde=True, ax=ax,
                      label='naive', color='m')
    ylim = ax.get_ylim()
    conversions = {'biases' : 'bias',
                   'slopes' : 'slope',
                   'rmses' : 'rmse'}
    if accessor == 'biases':
        x = experiments['causal']['bias']
        xnaive = experiments['reality']['bias']
    elif accessor == 'slopes':
        x = experiments['causal']['slope']
        xnaive = experiments['reality']['slope']
    elif accessor == 'rmses':
        x = experiments['causal']['rmse']
        xnaive = experiments['reality']['rmse']
    else:
        x = np.nan
        xnaive = np.nan
    x = experiments['causal'][conversions[accessor]]
    ax.plot(f([x, x]), ylim, label='\"truth\"')
    x = experiments['reality'][conversions[accessor]]
    ax.plot(f([x, x]), ylim)
    if extra_experiment:
        x = experiments[extra_experiment][conversions[accessor]]
        ax.plot(f([x, x]), ylim)
    plt.legend()
    ax.set_xlabel(accessor)
    return

hist_plot(experiments)

hist_plot(experiments, accessor='slopes')

for exp in experiment_names:
    hist_plot(experiments, accessor='slopes', extra_experiment=exp)
    hist_plot(experiments, accessor='biases',extra_experiment=exp)
    hist_plot(experiments, accessor='rmses',extra_experiment=exp)

scatter_plot(experiments, data=TRAIN)
plt.show()

# below is useful for testing if we are actually asumptoting to a
# "specificaiton error" as sample size increases

def fit_model(n):
    """Fit a model to a subsample of TRAIN with sample size N

Useful for testing if we are tryuly asymptoting our sampling error and estimate."""
    sample = TRAIN.sample(n=n, replace=True, random_state=0)
    model = LinearRegression()
    model.fit(X=prep_x_data(sample.SM), y=sample.ET)
    return model

samples = [10, 100, 500, 1000, 2000, 4000, 5000, 6000, TRAIN.shape[0]]
models = list(map(fit_model, samples))
predictions = list(map(lambda m: m.predict(prep_x_data(TEST_DF.SM)), models))
rmses = list(map(lambda p: rmse(TEST_DF.ET, p), predictions))
biases = list(map(lambda p: bias(TEST_DF.ET, p), predictions))

plt.figure()
plt.plot(samples, [m.coef_ for m in models])
plt.xlabel('n samples')
plt.ylabel('slope coef')

plt.figure()
plt.plot(samples,rmses)
plt.xlabel('n samples')
plt.ylabel('rmse')

plt.figure()
plt.plot(samples,biases)
plt.xlabel('n samples')
plt.ylabel('biases')

plt.show()
