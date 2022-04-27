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

# TODO: just put in one big dataframe and use gorupby apply, etc.?

data_dir = "./data"

def load_experiment(site, name):
    """Load experiment cooresponding to SITE and NAME."""
    df = pd.read_csv("%s/%s-%s.csv" % (data_dir, site, name))
    return df[(~np.isnan(df.ET)) & (df.ET >= 0.0)]

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
        xs = np.array([[sm_neg], [sm_0], [sm_pos]])
        ys = np.array([et_neg, et_0, et_pos])
        m.fit(X=xs, y=ys)
        df_out['slope'] = float(m.coef_)
        df_out['sum_squared_error'] = ((ys - m.predict(xs))**2).sum()
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
    shape0 = _df.shape[0]
    _df = _df[(~np.isnan(_df.slope)) & (_df['sum_squared_error'] <= 100.0)]
    print("Fraction of obs removed: %f\n" % (float(shape0 - _df.shape[0])/shape0))
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

def add_linear_regression(model, ax, name):
    """Add a fitted line from MODEL to AX, , and label it with NAME."""
    xlim = np.array(ax.get_xlim()).reshape(-1, 1)
    ax.plot(np.squeeze(xlim), np.squeeze(model.predict(xlim)),
            label=name)
    ax.set_xlim(np.squeeze(xlim))
    return ax

def scatter_plot(experiments, title=''):
    """Return a two-panel scatter plot of DATA
with regression fits overlaid"""
    fig = plt.figure()
    fig.set_figwidth(fig.get_figwidth()*2.0)
    axs = fig.subplots(nrows=1, ncols=2)
    for (ax, scatter_name) in zip(axs, experiments.keys()):
        ax = sns.scatterplot(data=experiments[scatter_name]['df'],
                                x='SM', y='ET', hue='slope', ax=ax)
        for (name, d) in experiments.items():
            ax = add_linear_regression(d['model'], ax, name)
        ax.set_title(scatter_name)
        ax.legend()
    plt.title(title)
    return

# below site order is low to increasing d/et dsm, but we could also cluster by similar cliamtes, etc.
SITE_ORDER = ['bergen', 'idar_oberstein', 'milano',  'quad_city',
              'lindenberg', 'las_vegas', 'kelowna', 'elko',
              'spokane', 'great_falls', 'riverton', 'flagstaff']

def slope_fit_plot(sites):
    """plto sum of squared error histogram for sites

TODO: make this one figure for all sites

use swarm plot or strip plot."""
    fig = plt.figure()
    fig.set_figwidth(fig.get_figwidth()*2.0)
    (ax0, ax1) = fig.subplots(nrows=1, ncols=2)
    dfs = c.deque()
    for d in sites.values():
        dfs.append(d['reality-slope']['df'])
    _df = pd.concat(dfs, ignore_index=True)
    ax0 = sns.stripplot(data=_df, x='sum_squared_error', y='site', ax=ax0,
                        order=SITE_ORDER)
    ax0.set_title('Reality')
    dfs = c.deque()
    for d in sites.values():
        dfs.append(d['randomized']['df'])
    _df = pd.concat(dfs, ignore_index=True)
    ax1 = sns.stripplot(data=_df, x='sum_squared_error', y='site', ax=ax1)
    ax1.set_title('Randomized')
    return

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
        f = open(pkl_path(site), 'wb')
        pickle.dump(experiments, f)
        f.close()
    return experiments

def normalize_y_axis(ax1, ax2):
    """make the limites of the y axis the same between AX1 and AX2"""
    ylim1 = ax1.get_ylim()
    ylim2 = ax2.get_ylim()
    lim= [min(ylim1[0], ylim2[0]), max(ylim1[1], ylim2[1])]
    ax1.set_ylim(lim)
    ax2.set_ylim(lim)
    return (ax1, ax2)

def slope_box_plot(sites, title=''):
    """make a box plot of the true vs naive slopes for each site"""
    fig = plt.figure()
    ax1 = fig.subplots(nrows=1, ncols=1)
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
    ax1 = sns.boxplot(x='site', y='dET/dSM', hue='slope type', data=df,
                      order=SITE_ORDER, ax=ax1)
    ax1.set_ylabel('dET/dSM (slope)')
    ax1.set_xlabel('Site')
    plt.legend()
    plt.title(title)
    return

def error_plot_absolute(sites, title=''):
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
    ax = sns.boxplot(x='site', y='dET/dSM error', hue='error type', data=df,
                     order=SITE_ORDER)
    ax.set_ylabel('dET/dSM absolute error')
    ax.set_xlabel('Site')
    plt.legend()
    plt.title(title)
    return

def error_plot(sites, title=''):
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
    ax = sns.boxplot(x='site', y='dET/dSM error', hue='error type', data=df,
                     order=SITE_ORDER)
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
# error_plot(sites)
error_plot_absolute(sites)

for (site, experiments) in sites.items():
    print("*****%s******" % site)
    print('max site: %f' % experiments['reality-slope']['df']['sum_squared_error'].max())
    print('max site: %f\n' % experiments['randomized']['df']['sum_squared_error'].max())

    # scatter_plot(experiments, title=site)


# sites where the slope is biased high (but these also usually
# have slightly lower error as well)
biased_high = {'las_vegas', 'elko'}

# sites were confounding counterintuitively /decreases/ error
# are these all the sites that have zero slopes?
confouding_decreases = {'las_vegas', 'elko', 'riverton', 'flagstaff'}

# have a portion of zero slope data
zero_slopes = {'elko', 'riverton', 'spokane', 'flagstaff', 'las_vegas'}

# bsk = cold semi-arid climate
bsk_sites = {'flagstaff', 'riverton', 'great_falls', 'elko', }

# hot desert climate
bwh_sites = {'las_vegas', }

# warm-summer humid continetnal climate
dfb_sites = {'spokane', 'kelowna', 'lindenberg'}

# hot-summer humid continental climate
dfa_sites = {'quad_city'}

# temperate ocean climate with no dry season
cfb_sites = {'idar-oberstein', 'bergen'}

# humid subtropical climate with no dry season
cfa_sites = {'milano'}

# WHAT IS GOING ON AT SPOKANE vs others?  spokane is kind of like
# greatfalls, but much more spread than elko, riverton, flagstaff and
# lasvegas in thenon-zero slope region. basically just shifted more
# arid than kelowna and lindenberg.

plt.show()

site_constants = pd.read_csv('%s/site-constants.csv' % data_dir)
site_constants.set_index('site', drop=False, inplace=True)
d = experiments['reality-slope']['df']
