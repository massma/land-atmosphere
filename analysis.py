import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import collections as c
import seaborn as sns
import random
import warnings
sns.set_theme()

#
# make figures, investigate weirdness at milano, bergen, and lindenberg

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

CLEAN_SITES = True
data_dir = "./data"

SITE_CONSTANTS = pd.read_csv('%s/site-constants.csv' % data_dir)
SITE_CONSTANTS.set_index('site', drop=False, inplace=True)
CONSTANT_KEYS = \
   ['C1sat',
    'C2ref',
    'CLa',
    'CLb',
    'CLc',
    'albedo',
    'cveg',
    'latt',
    'wfc',
    'wsat',
    'wwilt',
    'z0h',
    'z0m']

# random state is used for plotting figures for NNEIGHBORS
RANDOM_STATE = np.random.RandomState(0)

# below site order is clusterd by similar climates
SITE_ORDER = ['bergen', 'idar_oberstein', 'lindenberg', 'milano', 'kelowna',  'quad_city',
              'spokane', 'flagstaff', 'elko', 'las_vegas', 'riverton', 'great_falls' ]

EXPERIMENT_NAMES = ['randomized', 'reality-slope']
NNEIGHBORS =\
    { 'quad_city' : 10, # checked, could be 20 (similar to idar_oberstein)
      'las_vegas' : 10, # checked
      'flagstaff' : 20, # checked
      'kelowna' : 10, # checked
      'great_falls' : 10, # checked
      'bergen' : 10, # checked
      'spokane' : 10, # checked
      'riverton' : 20, # checked
      'elko' : 20, # checked
      'lindenberg' : 10, # checked
      'idar_oberstein' : 10, # checked, could be 20
      'milano' : 10, # checked
     }


NNEIGHBOR_TEST = np.concatenate(([5], np.arange(10, 110, 10)))

def load_experiment(site, name):
    """Load experiment cooresponding to SITE and NAME."""
    df = pd.read_csv("%s/%s-%s.csv" % (data_dir, site, name))
    return df[(~np.isnan(df.ET)) & (df.ET >= 0.0) & (df.tstart <= 8.0)]

def prep_x_data(ds):
    """Take a dataseries DS and make it into the form needed by scikit fit."""
    return ds.to_numpy().reshape(-1, 1)

def add_linear_regression(model, ax, name):
    """Add a fitted line from MODEL to AX, , and label it with NAME."""
    xlim = np.array(ax.get_xlim()).reshape(-1, 1)
    ax.plot(np.squeeze(xlim), np.squeeze(model.predict(xlim)),
            label=name)
    ax.set_xlim(np.squeeze(xlim))
    return ax

FIG = plt.figure()

def true_slope(_df, site, name):
    """Meant to be called on groupby(['year', 'doy']), wtih SITE and experiment NAME as argument"""
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
        df_out['backward_difference'] = (et_0 - et_neg) / (sm_0 - sm_neg)
        df_out['forward_difference'] = (et_pos - et_0) / (sm_pos - sm_0)
        df_out['centered_difference'] = (et_pos - et_neg) / (sm_pos - sm_neg)
        df_out['min_slope'] = np.amin([df_out['forward_difference'],
                                       df_out['backward_difference'],
                                       df_out['centered_difference'],
                                       df_out['slope']])
        df_out['max_slope'] = np.amax([df_out['forward_difference'],
                                       df_out['backward_difference'],
                                       df_out['centered_difference'],
                                       df_out['slope']])
        df_out['et_neg'] = et_neg
        df_out['sm_neg'] = sm_neg
        df_out['et_pos'] = et_pos
        df_out['sm_pos'] = sm_pos
        df_out['slope_model'] = m
        return df_out

def true_fit_figure(ds, site, name):
    """plot the fit of a data series DS, for SITE and experiment NAME."""
    FIG.clf(True)
    ax = FIG.add_subplot(111)
    ax.plot([ds['sm_neg'], ds['SM'], ds['sm_pos']],
            [ds['et_neg'], ds['ET'], ds['et_pos']], 'k*')
    ax.set_title('Sum squared error: %f' % ds['sum_squared_error'])
    add_linear_regression(ds['slope_model'], ax, '')
    plt.savefig('diagnostic_figures/%s-%s/%d-%03d_derivative_fit.png'
                % (site, name,  ds.year, ds.doy))
    return

def plot_neighbors(df, site):
    plt.figure()
    n_neighbors_test = NNEIGHBOR_TEST
    plt.plot(n_neighbors_test, [neighbor_error_f(n, df) for n in n_neighbors_test],'k*')
    plt.ylabel('Average squared error per point')
    plt.xlabel('N neighbors')
    plt.title('%s, regular' % site)
    plt.show()
    return True

ATM_KEYS = {'theta', 'advtheta', 'q', 'advq', 'cc', 'ws', 'h', 'day'}
LAND_KEYS = {'T2', 'Tsoil', 'Ts', 'LAI', 'SM'}
CONTROL_KEYS = ATM_KEYS.union(LAND_KEYS)

def normalized_key(key):
    """return the normalized key from a standard key"""
    return '%s_normalized' % key

def normalized_prep(_df):
    """alternate Step 1 in my causal analysis that is more "mainstream":

normalize each variable"""
    for key in CONTROL_KEYS:
        _df[normalized_key(key)] = (_df[key] - _df[key].mean())/_df[key].std()
    return _df

def fit_neighbors(_df, keys, classification_key, key_mapper, n_neighbors):
    """Build neighbors based on values assigned to KEY, and assign
classification to CLASSIFICATION_KEY.

    KEY_MAPPER is a function that maps a key to the metric by which we are neighboring.

    """
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    xs = _df[list(map(key_mapper, keys))].values
    neigh = neigh.fit(xs)
    _df[classification_key] = list(neigh.kneighbors(xs, return_distance=False))
    return _df

def model_neighbors(neighbors, _df):
    """intended to be called on neibors, fit linear model on _DF and return slope and count"""
    df = _df.iloc[neighbors, :]
    m = LinearRegression()
    m.fit(X=prep_x_data(df.SM), y=df.ET)
    return m

def error_neighbors(model, neighbors, _df):
    """Score model using neighbors in _df.

Metric is average squared error (so it doesn't grow eith neighbor size.

Also neighbors should be held out data."""
    df = _df.iloc[neighbors, :]
    predicts = np.squeeze(model.predict(X=prep_x_data(df.SM)))
    return np.average((predicts - df.ET)**2)

def neighbor_error_f(n_neighbors, df):
    """Return average of squared error for nneighbor and df"""
    if ((n_neighbors * 0.8) % 1) != 0.0:
        raise ValueError('n_neighbors not divisible by 5 for testing!')
    df = df.copy(deep=True)
    df = fit_neighbors(normalized_prep(df), CONTROL_KEYS,
                       'neighbors', normalized_key,
                       n_neighbors)
    n_samples = 5
    averages = c.deque()
    for neighbors in df['neighbors'].values:
        sum_averages = 0
        for _i in range(n_samples):
            (train, test) = train_test_split(neighbors,
                                             random_state=RANDOM_STATE,
                                             train_size=0.8)
            m = model_neighbors(train, df)
            sum_averages = sum_averages + error_neighbors(m, test, df)
        averages.append((sum_averages/float(n_samples)))
    return np.average(averages)

def neighbor_effect(df, n_neighbors):
    """Perform 4 steps in my causal estimatation method:
1. Normalize each confounder we're adjusting for. (this is more
   accepted method, but makes less snes to me for "nearness"
2. Calculate a k nearest neighbors classifier
3. Calcualte a local slope based on each k nearest neighbors.
    """
    df = df.copy(deep=True)
    (df) =\
        fit_neighbors(normalized_prep(df), CONTROL_KEYS,
                      'neighbors', normalized_key,
                      n_neighbors)
    df['neighbor_effect_model'] = [model_neighbors(neighbors, df) for neighbors in
                                   df['neighbors']]
    df['neighbor_slope'] = [float(m.coef_) for m in df['neighbor_effect_model']]

    return df

def naive_regression(_df):
    """Calulcate a naive regression on _df. Return the model.

"""
    m = LinearRegression()
    m.fit(X=prep_x_data(_df.SM), y=_df.ET)
    return m

def expert_naive_regression(_df, site):
    """Calulcate a naive regression on _df but using expert guidance to
account for areas we know dET/dSM = 0 (SM < wilt, SM > wilt).

Basically set the slope equal to zero for all of those areas.
"""
    wwilt = SITE_CONSTANTS.loc[site, 'wwilt']
    wfc = SITE_CONSTANTS.loc[site, 'wfc']
    full_size = float(_df.shape[0])
    _df = _df[(_df.SM > wwilt) & (_df.SM < wfc)]
    m = LinearRegression()
    m.fit(X=prep_x_data(_df.SM), y=_df.ET)
    return (float(m.coef_) * float(_df.shape[0]) / full_size)

def expert_df(_df, site):
    """return an expert guided _df"""
    wwilt = SITE_CONSTANTS.loc[site, 'wwilt']
    wfc = SITE_CONSTANTS.loc[site, 'wfc']
    _df = _df[(_df.SM > wwilt) & (_df.SM < wfc)]
    return _df

def load_calculate_truth(site, name):
    """lad data and calculate the true slope for SITE and experiment NAME.

Returns a dicntionary with data and slopes."""
    df = load_experiment(site, name)
    df['ws'] = np.sqrt(df.u**2 + df.v**2)
    os.makedirs('diagnostic_figures/%s-%s' % (site, name), exist_ok=True)
    df = df.groupby(['year', 'doy']).apply(lambda _df: true_slope(_df, site, name))
    shape0 = df.shape[0]
    df = df[(~np.isnan(df.slope))]
    print("Fraction of obs removed nan slopes: %f\n" % (float(shape0 - df.shape[0])/shape0))
    return df

def add_neighbor_fit(df, site):
    """Add neighbor fit to site"""
    n_neighbors = NNEIGHBORS[site]
    print("N neighbors for site %s: %d\n" % (site, n_neighbors))
    df = neighbor_effect(df, n_neighbors)
    return df

def rmse(truth, prediction):
    """Return the RMSE between TRUTH (dataseries) and PREDICTION (np array)"""
    return np.sqrt(np.average((prediction-prep_x_data(truth))**2))

def bias(truth, prediction):
    """Return the bias between TRUTH (dataseries and PREDICTION (np array)"""
    return np.average(prediction-prep_x_data(truth))

def cross_product(f, xs1, xs2):
    """Apply f to cross of xs1 and xs2"""
    return np.array([f(x1, x2) for x1 in xs1 for x2 in xs2])

def scatter_plot(experiments, title=''):
    """Return a two-panel scatter plot of DATA
with regression fits overlaid"""
    fig = plt.figure()
    fig.set_figwidth(fig.get_figwidth()*2.0)
    axs = fig.subplots(nrows=1, ncols=2)
    for (ax, scatter_name) in zip(axs, experiments.keys()):
        ax = sns.scatterplot(data=experiments[scatter_name],
                             x='SM', y='ET', hue='slope', ax=ax)
        ax.set_title(scatter_name)
        ax.legend()
    normalize_y_axis(*axs)
    plt.title(title)
    return

CONCATS = dict()
def concat_experiment(key):
    """Concat all experiments given by KEY ('reality-slope' or 'randomized').

This does a crude memoization to CONCATS, which assumes that sites is
always the same input. (a very bad idea)

    """
    if key in CONCATS:
        return CONCATS[key]
    else:
        dfs = c.deque()
        for site in SITE_ORDER:
            dfs.append(SITES[site][key]['df'])
        df = pd.concat(dfs, ignore_index=True)
        CONCATS[key] = df
        return df

def slope_fit_plot():
    """plto sum of squared error histogram for sites

use swarm plot or strip plot."""
    fig = plt.figure()
    fig.set_figwidth(fig.get_figwidth()*2.0)
    (ax0, ax1) = fig.subplots(nrows=1, ncols=2)
    _df = concat_experiment('reality-slope')
    ax0 = sns.stripplot(data=_df, x='sum_squared_error', y='site', ax=ax0,
                        order=SITE_ORDER)
    ax0.set_title('Reality')
    _df = concat_experiment('randomized')
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
        experiments = dict()
        for name in EXPERIMENT_NAMES:
            _df = load_calculate_truth(site, name)
            if name == 'reality-slope':
                _df = add_neighbor_fit(_df, site)
            experiments[name] = _df
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

def slope_box_plot(title=''):
    """make a box plot of the true vs naive slopes for each site"""
    fig = plt.figure()
    ax1 = fig.subplots(nrows=1, ncols=1)
    dfs = c.deque()
    for (site, experiments) in SITES.items():
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

def slope_adjustment_box_plot(title=''):
    """make a box plot of the true vs adjusted slopes for each site"""
    fig = plt.figure()
    ax1 = fig.subplots(nrows=1, ncols=1)
    dfs = c.deque()
    for (site, experiments) in SITES.items():
        d = experiments['reality-slope']
        _df = pd.DataFrame(d['naive_slopes'], columns=['dET/dSM'])
        _df['slope type'] = 'naive'
        _df['site'] = site
        dfs.append(_df)

        _df = pd.DataFrame(d['neighbor_slopes'],
                           columns=['dET/dSM'])
        _df['slope type'] = 'adjusted'
        _df['site'] = site
        dfs.append(_df)

        # if site in ['elko', 'las_vegas']:
        _df = pd.DataFrame(d['expert_neighbor_slopes'],
                           columns=['dET/dSM'])
        _df['slope type'] = 'adjusted\nw/ expert'
        _df['site'] = site
        dfs.append(_df)

        _df = pd.DataFrame(d['true_slopes'],
                           columns=['dET/dSM'])
        _df['slope type'] = 'truth'
        _df['site'] = site
        dfs.append(_df)
    df = pd.concat(dfs, ignore_index=True)
    ax1 = sns.boxplot(x='site', y='dET/dSM', hue='slope type', data=df,
                      order=SITE_ORDER, ax=ax1,
                      hue_order=['naive', 'adjusted', 'adjusted\nw/ expert', 'truth'])
    ax1.set_ylabel('dET/dSM (slope)')
    ax1.set_xlabel('Site')
    plt.legend()
    plt.title(title)
    return

def error_plot_absolute(title=''):
    """make a box plot of error due to confounding and specification"""
    fig, ax = plt.subplots()
    dfs = c.deque()
    for (site, experiments) in SITES.items():
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



def error_adjustment_plot_absolute(title=''):
    """make a box plot of error due to confounding and specification"""
    fig, ax = plt.subplots()
    dfs = c.deque()
    for (site, experiments) in SITES.items():
        d = experiments['reality-slope']
        _df = pd.DataFrame(np.absolute(d['naive_errors']),
                           columns=['dET/dSM error'])
        _df['error type'] = 'naive'
        _df['site'] = site
        dfs.append(_df)


        _df = pd.DataFrame(np.absolute(d['neighbor_errors']),
                           columns=['dET/dSM error'])
        _df['error type'] = 'adjusted'
        _df['site'] = site
        dfs.append(_df)

        _df = pd.DataFrame(np.absolute(d['expert_neighbor_errors']),
                           columns=['dET/dSM error'])
        _df['error type'] = 'adjusted\nw/ expert'
        _df['site'] = site
        dfs.append(_df)

    df = pd.concat(dfs, ignore_index=True)
    ax = sns.boxplot(x='site', y='dET/dSM error', hue='error type', data=df,
                     order=SITE_ORDER,
                     hue_order=['naive', 'adjusted', 'adjusted\nw/ expert']
                     )
    ax.set_ylabel('dET/dSM absolute error')
    ax.set_xlabel('Site')
    plt.legend()
    plt.title(title)
    return

def error_plot(title=''):
    """make a box plot of error due to confounding and specification"""
    fig, ax = plt.subplots()
    dfs = c.deque()
    for (site, experiments) in SITES.items():
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

SITES = dict()


for site in stations.keys():
    if CLEAN_SITES and os.path.exists(pkl_path(site)):
        os.remove(pkl_path(site))
    print('Working on %s\n' % site)
    SITES[site] = site_analysis(site)

for site in stations.keys():
    for exp in EXPERIMENT_NAMES:
        print("%s, %s size: %d" % (site, exp, SITES[site][exp].shape[0]))

for (site, experiments) in SITES.items():

    scatter_plot(experiments, title=site)

plt.show()

def fraction_wilt(site):
    """Return the fraction of obs that are below wilting point for SITE"""
    _df = SITES[site]['reality-slope']['df']
    return (float(_df[_df.SM <
                      float(SITE_CONSTANTS.loc[site, 'wwilt'])].shape[0])
            / float(_df.shape[0]))

def fraction_fc(site):
    """Return the fraction of obs that are above field capcaity for SITE"""
    _df = SITES[site]['reality-slope']['df']
    return (float(_df[_df.SM >
                      float(SITE_CONSTANTS.loc[site, 'wfc'])].shape[0])
            / float(_df.shape[0]))

for site in SITE_ORDER:
    print('%s fraction below wilt : %f\n'
          % (site, fraction_wilt(site)))
    print('%s fraction above fc : %f\n'
          % (site, fraction_fc(site)))

# sites where the slope is biased high (but these also usually
# have slightly lower error as well)
biased_high = {'las_vegas', 'elko'}

# sites were confounding counterintuitively /decreases/ error
# are these all the sites that have zero slopes?
confouding_decreases = {'las_vegas', 'elko', 'riverton', 'flagstaff'}

# have a portion of zero slope data
zero_slopes = {'elko', 'riverton', 'spokane', 'flagstaff', 'las_vegas'}


# temperate ocean climate with no dry season
cfb_sites = {'idar_oberstein', 'bergen', 'lindenberg'}

# humid subtropical climate with no dry season
cfa_sites = {'milano'}

# subartic climate
dfc_sites = {'kelowna'}

# hot-summer humid continental climate
dfa_sites = {'quad_city'}

# mediterannean-influenced warm-summer humid continental climate
dsb_sites = {'elko'}

# warm-summer mediterannean climate
csb = {'spokane', 'flagstaff'}

# cold desert climate
bwk_climate = {'riverton', 'las_vegas'}

# cold semi-arid climate
bsk_sites = {'great_falls'}

SITE_CLIMATES = \
    { 'bergen' : 'Cfb',
      'idar_oberstein' : 'Cfb',
      'lindenberg' : 'Cfb',
      'milano' : 'Cfa',
      'kelowna' : 'Dfc',
      'quad_city' : 'Dfa',
      'spokane' : 'Csb',
      'flagstaff' : 'Csb',
      'elko' : 'Dsb',
      'las_vegas' : 'Bwk',
      'riverton' : 'Bwk',
      'great_falls' : 'Bsk'}

# spokane and flagstaff; great falls and riverton are nice comparisons

# WHAT IS GOING ON AT SPOKANE vs others?  spokane is kind of like
# greatfalls, but much more spread than elko, riverton, flagstaff and
# lasvegas in thenon-zero slope region. basically just shifted more
# arid than kelowna and lindenberg.


STATISTICS_KEYS = \
   [ 'SM', 'ET', 'slope', 'T2', 'Tsoil', 'Ts', 'theta', 'q', 'LAI', 'cc', 'h', 'tstart', 'day']

# diagnostics: rh

EPS = 0.622

# thermo funcitons for calculating RH
def e_s(t_a):
    """calculates e_s in Pa, from shuttleworth equation 2.17.
    t_a is in kelvin"""
    t_c = t_a - 273.15
    return 610.8 * np.exp(17.27 * t_c / (237.3 + t_c))

def specific_humidity(e_s, p):
    """slide 9, lecture 2; p and e_s must be same units"""
    return EPS * e_s / (p - (1.0 - EPS) * e_s)

def e_from_q(q, p):
    """invert `specific_humidity'"""
    return q * p / (EPS + (1 - EPS) * q)

def rh_from_t_a_q_p(t_a, q, p):
    """calcualte rh from t_a (in kelvin), q (unitless), and p (Pa)"""
    return np.minimum(1.0, np.maximum(0.0, (e_from_q(q, p)) / e_s(t_a)))

def t_from_theta_p(theta, p):
    """inverts `theta_from_t_p'. Theta in kelvin, pressure in pascal."""
    return theta * (p / 100000.0) ** (2.0 / 7.0)

def site_comparison_figures():
    """Compare each site in a figure. For now, generate a spearate figure for each site"""
    for key in CONSTANT_KEYS:
        fig = plt.figure()
        ax = fig.subplots(nrows=1, ncols=1)
        sns.stripplot(x=SITE_ORDER,
                      y=[SITE_CONSTANTS.loc[site, key]
                         for site in SITE_ORDER])
        ax.set_title(key)
        ax.set_ylabel(key)
        ax.set_xlabel(key)
    df = concat_experiment('reality-slope')
    df['rh'] = rh_from_t_a_q_p(df['theta'],
                               df['q'] / 1000.0,
                               df['pressure']*100.0)
    df['ws'] = np.sqrt(df['u']**2 + df['v']**2)
    keys = ['ws', 'rh']
    keys.extend(STATISTICS_KEYS)
    for key in keys:
        fig = plt.figure()
        ax = fig.subplots(nrows=1, ncols=1)
        sns.boxplot(x='site', y=key, data=df, order=SITE_ORDER)
        ax.set_title(key)
    return

def confounding_error(site):
    """Return the difference in absolute error between naiv-slope and randomized
    for SITE"""
    err = np.absolute(SITES[site]['reality-slope']['naive_error']) -\
          np.absolute(SITES[site]['randomized']['naive_error'])
    return err

def final_site_comparison_figures():
    """Make a subset of final sites comparison plots. THese should be publication ready.

Currently, this subset is: (we really probably just wantt thist o be
SM in the end, and part of the other panel fig with slope estiamtes
and error

SM (w/wfc, and wwilt) (wsat is just used fo rheat transfer)
et
slope
lai
theta
rh
cc

    """
    df = concat_experiment('reality-slope')
    df['rh'] = rh_from_t_a_q_p(df['theta'],
                               df['q'] / 1000.0,
                               df['pressure']*100.0)
    df['climate'] = [SITE_CLIMATES[site] for site in df.site]
    fig = plt.figure()
    ax = fig.subplots(nrows=1, ncols=1)
    ax = sns.boxplot(x='site', y='SM', data=df, order=SITE_ORDER, ax=ax,
                     hue='climate', dodge=False)
    ax = sns.stripplot(x='site', y='wwilt', data=SITE_CONSTANTS,
                       ax=ax, order=SITE_ORDER, color='k')
    ax = sns.stripplot(x='site', y='wfc', data=SITE_CONSTANTS,
                       ax=ax, order=SITE_ORDER, color='m')
    ax.set_title('SM')
    ax.legend([], [], frameon=False)

    for key in ['ET', 'slope', 'LAI', 'theta', 'rh', 'cc']:
        fig = plt.figure()
        ax = fig.subplots(nrows=1, ncols=1)
        ax = sns.boxplot(x='site', y=key, data=df, order=SITE_ORDER, ax=ax,
                         hue='climate', dodge=False)
        ax.set_title(key)
        ax.legend([], [], frameon=False)
    return True

df = concat_experiment('reality-slope')


# make this a scatter with site legend?
#
# fig = plt.figure()
# ax = fig.subplots(nrows=1, ncols=1)
# ax.plot([fraction_wilt(site) for site in SITE_ORDER],
#          [confounding_error(site) for site in SITE_ORDER],
#          'k.')
# ax.set_ylabel('confounding error')
# ax.set_xlabel('fraction of obs below wilting point')
final_site_comparison_figures()
slope_box_plot()
error_plot_absolute()
slope_adjustment_box_plot()
error_adjustment_plot_absolute()
plt.show()

# hypthesis for spokane and great falls: less fraction sub wwilt than other sites
# (and higher latitude/more seasonal cycle)
