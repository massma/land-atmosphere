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
from sklearn.cluster import KMeans


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

NSAMPLE = 100 # for bootstrap

# causal effect functions

ATM_KEYS = {'theta', 'advtheta', 'q', 'advq', 'cc', 'u', 'v', 'h', 'pressure', 'day'}
LAND_KEYS = {'T2', 'Tsoil', 'Ts', 'LAI'}
CONTROL_KEYS = ATM_KEYS.union(LAND_KEYS)
CAUSAL_CONTROLS = {'atm', 'land', 'all'}

def rank_key(key):
    """return the rank key from a standard key"""
    return '%s_rank' % key

def rank_prep(_df):
    """Step 1 in my causal analysis: rank each variable"""
    for key in CONTROL_KEYS:
        _df[rank_key(key)] = _df[key].rank()
    return _df

POINTS_PER_CLUSTER = 50
def classify(_df, keys, classification_key):
    """Classify each point in _DF based on rank values cooresponding to
    KEY, and assign classification to CLASSIFICATION_KEY
    """
    n_clusters = np.int(np.floor(float(_df.shape[0]) / float(POINTS_PER_CLUSTER)))
    max_iter=1000
    model = KMeans(n_clusters=n_clusters, max_iter=max_iter, algorithm="full",
                   random_state=0)
    xs = _df[list(map(rank_key, keys))].values
    model.fit(xs)
    if model.n_iter_ == max_iter:
        raise Exception('ERORR: max iterations reached when iftting model on %s' % classification_key)
    _df[classification_key] = model.predict(xs)
    return _df

def assign_clusters(_df):
    """Add all clusters to _DF"""
    _df = classify(_df, ATM_KEYS, 'atm_cluster')
    _df = classify(_df, LAND_KEYS, 'land_cluster')
    _df = classify(_df, CONTROL_KEYS, 'all_cluster')
    return _df

def model_cluster(_df):
    """intended to be called on groupby, fit linear model on _DF and return slope and count"""
    m = LinearRegression()
    m.fit(X=prep_x_data(_df.SM), y=_df.ET)
    return pd.Series({'slope' : np.float(m.coef_),
                      'count' : np.float(_df.shape[0])})


def calculate_effect(grouped):
    """Calulcate causal effect from GROUPED: a dataframe of slopes and counts"""
    return (grouped['slope'] * grouped['count']).sum()/grouped['count'].sum()

def prep_effect(_df):
    """Perform the first 2 steps in my causal estimation method:

1. Rank each confounder we're adjusting for.
2. Assign each point a cluster based on rankings ("match").
"""
    return assign_clusters(rank_prep(_df))

def estimate_effect(_df, confounder_set):
    """Perform last 2 steps in my causal estimatation method:

3. Group by each cluster and calcualte a slope.
4. Calculate the average slope, weighted by the number of points in each cluster.
"""
    if confounder_set not in CAUSAL_CONTROLS:
        raise ValueError('ERROR: estimate_effect called with an \
 unknown or unimplemented confounder set: %s' % confounder_set)
    return calculate_effect(_df.groupby('%s_cluster' % confounder_set)\
                            .apply(model_cluster))

def fit_models(experiments):
    """fit models and calculate slopes for all EXPERIMENTS"""
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
        d['slope'] = model.coef_
        d['slopes'] = np.array([float(m.coef_) for m in models])
        if model_string == 'reality':
            d['df'] = prep_effect(d['df'])
            d['samples'] = [prep_effect(_df) for _df in d['samples']]
            d['true-slope'] = d['df'].slope.mean()
            d['true-slopes'] = np.array([x.slope.mean() for x in samples])
    d = experiments['reality']
    for key in CAUSAL_CONTROLS:
        experiments['%s_effect' % key] =\
            { 'slope' : estimate_effect(d['df'], key),
              'slopes' : np.array([estimate_effect(_df, key)
                                   for _df in d['samples']])}

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
        d['mean-slope'] = np.average(d['slopes'])
        d['std-slope'] = np.std(d['slopes'])
        cross=np.array([estimate - truth
                        for estimate in d['slopes']
                        for truth in experiments['reality']['true-slopes']])
        d['mean-bias'] = np.average(cross)
        d['std-bias'] = np.std(cross)
    return experiments

experiments = model_diagnostics(experiments)

def scatter_plot(experiments, experiment='randomized', title=''):
    """return a scatter plot of DATA with regression fits overlaid"""
    facet = sns.relplot(data=experiments[experiment]['df'], x='SM', y='ET')
    ax = facet.ax
    xlim = np.array(ax.get_xlim()).reshape(-1, 1)
    for (name, d) in experiments.items():
        try:
            ax.plot(np.squeeze(xlim), np.squeeze(d['model'].predict(xlim)), label=name)
        except KeyError: # causal estimtes don't have a model; just a slope
            'okay'
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
    _df = pd.DataFrame(experiments['reality']['true-slopes'],
                       columns=['dET/dSM'])
    _df['name'] = 'truth'
    dfs.append(_df)
    df = pd.concat(dfs, ignore_index=True)
    ax = sns.boxplot(x='name', y='dET/dSM', data=df)
    ax.set_ylabel('dET/dSM (slope)')
    ax.set_xlabel('Experiment')
    return

box_plot(experiments)

scatter_plot(experiments)

plt.show()
