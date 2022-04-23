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

def load_experiment(site, name):
    """Load experiment cooresponding to SITE and NAME."""
    df = pd.read_csv("%s/%s-%s.csv" % (data_dir, site, name))
    return df[(~np.isnan(df.ET)) & (df.ET <= 1000.0) & (df.ET >= 0.0)]

def prep_x_data(ds):
    """Take a dataseries DS and make it into the form needed by scikit fit."""
    return ds.to_numpy().reshape(-1, 1)

def true_slope(_df):
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


NSAMPLE = 50 # for bootstrap

ATM_KEYS = {'theta', 'advtheta', 'q', 'advq', 'cc', 'u', 'v', 'h', 'pressure', 'day'}
LAND_KEYS = {'T2', 'Tsoil', 'Ts', 'LAI'} # should we include SM here?
                                         # it taints the idea of
                                         # matching, but makes sense
                                         # with the idea of piecewise
                                         # linear
CONTROL_KEYS = ATM_KEYS.union(LAND_KEYS)

def rank_key(key):
    """return the rank key from a standard key"""
    return '%s_rank' % key

def rank_prep(_df):
    """Step 1 in my causal analysis: rank each variable"""
    for key in CONTROL_KEYS:
        _df[rank_key(key)] = _df[key].rank()
    return _df

# average number of points we want per a cluster
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
    _df = classify(_df, CONTROL_KEYS, 'cluster')
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

def estimate_effect(_df):
    """Perform last 2 steps in my causal estimatation method:

3. Group by each cluster and calcualte a slope.
4. Calculate the average slope, weighted by the number of points in each cluster.
"""
    return calculate_effect(_df.groupby('cluster')\
                            .apply(model_cluster))

def fit_models(d):
    """fit models and calculate slopes for D.

D is a dictionary with a 'df' key. D will be mutated
along the way, adding slopes, etc."""
    _df = d['df'].groupby(['year', 'doy']).apply(true_slope)
    _df = _df[~np.isnan(_df.slope)]
    d['df'] = _df
    samples = [d['df'].sample(n=d['df'].shape[0],
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
    d['naive_slope'] = model.coef_
    d['naive_slopes'] = np.array([float(m.coef_) for m in models])

    d['df'] = prep_effect(_df)
    d['samples'] = [prep_effect(_df) for _df in d['samples']]
    d['true_slope'] = d['df'].slope.mean()
    d['true_slopes'] = np.array([x.slope.mean() for x in samples])

    d['causal_slope'] = estimate_effect(d['df']),
    d['causal_slopes'] = np.array([estimate_effect(_df)
                                   for _df in d['samples']])

    return d


def rmse(truth, prediction):
    """Return the RMSE between TRUTH (dataseries) and PREDICTION (np array)"""
    return np.sqrt(np.average((prediction-prep_x_data(truth))**2))

def bias(truth, prediction):
    """Return the bias between TRUTH (dataseries and PREDICTION (np array)"""
    return np.average(prediction-prep_x_data(truth))

def cross_product(f, xs1, xs2):
    """Apply f to cross of xs1 and xs2"""
    return np.array([f(x1, x2) for x1 in xs1 for x2 in xs2])

def model_diagnostics(d):
    """Generate model diangostics for each dictionary (D) of an experiment.

Mutates dictionary to add diagnostic terms.
"""
    d['naive_error'] = d['naive_slope'] - d['true_slope']
    d['naive_errors'] = cross_product((lambda x1, x2: x1 - x2),
                                      d['naive_slopes'], d['true_slopes'])
    return d

# def scatter_plot(experiments, experiment='randomized', title=''):
#     """return a scatter plot of DATA with regression fits overlaid"""
#     facet = sns.relplot(data=experiments[experiment]['df'], x='SM', y='ET')
#     ax = facet.ax
#     xlim = np.array(ax.get_xlim()).reshape(-1, 1)
#     for (name, d) in experiments.items():
#         try:
#             ax.plot(np.squeeze(xlim), np.squeeze(d['model'].predict(xlim)), label=name)
#         except KeyError: # causal estimtes don't have a model; just a slope
#             'okay'
#     mean_slope = experiments['reality']['true-slope']
#     f = lambda x: experiments[experiment]['model'].intercept_ + mean_slope * x
#     ax.plot(np.squeeze(xlim), list(map(f, np.squeeze(xlim))), label='truth')
#     plt.title(title)
#     plt.legend()
#     return


def pkl_path(site):
    """Return the path to a pickle file for SITE"""
    return '%s/%s.pkl' % (data_dir, site)

def load_pickled_experiments(site):
    """Load pickled experiments for SITE"""
    f = open(pkl_path(site), 'rb')
    experiments = pickle.load(f)
    f.close()
    return experiments

def site_analysis(site):
    """Execute all analysis on SITE

As a side effect, may write a pickle file to data/SITE.pkl"""

    if os.path.exists(pkl_path(site)):
        experiments = load_pickled_experiments(site)
    else:
        experiment_names = ['randomized',
                            'reality-slope']
        experiments = dict()
        for name in experiment_names:
            experiments[name] = model_diagnostics(
                fit_models({'df' : load_experiment(site, name)}))
        experiments['confounding_error'] =\
             experiments['reality-slope']['naive_error'] - experiments['randomized']['naive_error']
        experiments['confounding_errors'] =\
             cross_product((lambda x1, x2: x1 - x2),
                           experiments['reality-slope']['naive_errors'],
                           experiments['randomized']['naive_errors'])
        experiments['confounding_absolute_error'] =\
             np.absolute(experiments['reality-slope']['naive_error']) -\
             np.absolute(experiments['randomized']['naive_error'])
        experiments['confounding_absolute_errors'] =\
             cross_product((lambda x1, x2: x1 - x2),
                           np.absolute(experiments['reality-slope']['naive_errors']),
                           np.absolute(experiments['randomized']['naive_errors']))

        f = open(pkl_path(site), 'wb')
        pickle.dump(experiments, f)
        f.close()
    return experiments


def slope_box_plot(sites, title=''):
    """make a box plot of the true vs naive slopes for each site"""
    fig, ax = plt.subplots()
    dfs = c.deque()
    for (site, experiments) in sites.items():
        d = experiments['reality-slope']
        _df = pd.DataFrame(d['naive_slopes'], columns=['dET/dSM'])
        _df['slope type'] = 'naive'
        _df['site'] = site
        dfs.append(_df)
        _df = pd.DataFrame(d['true_slopes'],
                           columns=['dET/dSM'])
        _df['slope type'] = 'truth'
        _df['site'] = site
        dfs.append(_df)
    df = pd.concat(dfs, ignore_index=True)
    ax = sns.boxplot(x='site', y='dET/dSM', hue='slope type', data=df)
    ax.set_ylabel('dET/dSM (slope)')
    ax.set_xlabel('Site')
    plt.legend()
    plt.title(title)
    return

def error_plot(sites, title=''):
    """make a box plot the due to confounding and specification for each site

this can take awhile."""
    fig, ax = plt.subplots()
    dfs = c.deque()
    for (site, experiments) in sites.items():
        _df = pd.DataFrame(np.absolute(experiments['randomized']['naive_errors']),
                           columns=['dET/dSM error'])
        _df['error type'] = 'specification'
        _df['site'] = site
        dfs.append(_df)
        _df = pd.DataFrame(np.absolute(experiments['confounding_errors']),
                           columns=['dET/dSM error'])
        _df['error type'] = 'confounding'
        _df['site'] = site
        dfs.append(_df)
    df = pd.concat(dfs, ignore_index=True)
    ax = sns.boxplot(x='site', y='dET/dSM error', hue='error type', data=df)
    ax.set_ylabel('dET/dSM absolute error')
    ax.set_xlabel('Site')
    plt.legend()
    plt.title(title)
    return

def error_plot_2_absolute(sites, title=''):
    """make a box plot of error due to confounding and specification"""
    fig, ax = plt.subplots()
    dfs = c.deque()
    for (site, experiments) in sites.items():
        _df = pd.DataFrame(np.absolute(experiments['randomized']['naive_errors']),
                           columns=['dET/dSM error'])
        _df['error type'] = 'specification'
        _df['site'] = site
        dfs.append(_df)
        _df = pd.DataFrame(np.absolute(experiments['reality-slope']['naive_errors']),
                           columns=['dET/dSM error'])
        _df['error type'] = 'specification & confounding'
        _df['site'] = site
        dfs.append(_df)
    df = pd.concat(dfs, ignore_index=True)
    ax = sns.boxplot(x='site', y='dET/dSM error', hue='error type', data=df)
    ax.set_ylabel('dET/dSM absolute error')
    ax.set_xlabel('Site')
    plt.legend()
    plt.title(title)
    return

def error_plot_2(sites, title=''):
    """make a box plot of error due to confounding and specification"""
    fig, ax = plt.subplots()
    dfs = c.deque()
    for (site, experiments) in sites.items():
        _df = pd.DataFrame(experiments['randomized']['naive_errors'],
                           columns=['dET/dSM error'])
        _df['error type'] = 'specification'
        _df['site'] = site
        dfs.append(_df)
        _df = pd.DataFrame(experiments['reality-slope']['naive_errors'],
                           columns=['dET/dSM error'])
        _df['error type'] = 'specification & confounding'
        _df['site'] = site
        dfs.append(_df)
    df = pd.concat(dfs, ignore_index=True)
    ax = sns.boxplot(x='site', y='dET/dSM error', hue='error type', data=df)
    ax.set_ylabel('dET/dSM error')
    ax.set_xlabel('Site')
    plt.legend()
    plt.title(title)
    return

f = open('%s/stations.pkl' % data_dir, 'rb')
# stations is a dictionary of {human readable site name : igra station ids}
stations = pickle.load(f)
f.close()

sites = dict()
CLEAN_SITES = False

for site in stations.keys():
    if CLEAN_SITES and os.path.exists(pkl_path(site)):
        os.remove(pkl_path(site))
    print('Working on %s\n' % site)
    sites[site] = site_analysis(site)

slope_box_plot(sites)
error_plot_2(sites)
error_plot_2_absolute(sites)

plt.show()
