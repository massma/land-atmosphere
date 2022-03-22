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

experiments = ["causal", "decorrelated", "reality"]

def load_experiment(name):
    "Load experiment NAME."
    df = pd.read_csv("%s/%s.csv" % (data_dir, name))
    return df[~np.isnan(df.ET)]

causal = load_experiment("causal")
decorrelated = load_experiment("decorrelated")
reality = load_experiment("reality")
subsampled = decorrelated.sample(n=reality.shape[0], random_state=1)
subsampled2 = decorrelated.sample(n=reality.shape[0], random_state=15)


naive_model = LinearRegression()
sample_limited_model = LinearRegression()
sample_limited_model2 = LinearRegression()
decorrelated_model = LinearRegression()

def prep_x_data(ds):
    """Take a dataseries DS and make it into the form needed by scikit fit."""
    return ds.to_numpy().reshape(-1, 1)


naive_model.fit(X=prep_x_data(reality.SM), y=reality.ET)
sample_limited_model.fit(X=prep_x_data(subsampled.SM), y=subsampled.ET)
sample_limited_model2.fit(X=prep_x_data(subsampled2.SM), y=subsampled2.ET)
decorrelated_model.fit(X=prep_x_data(decorrelated.SM), y=decorrelated.ET)

def rmse(truth, prediction):
    """Return the RMSE between TRUTH (dataseries) and PREDICTION (np array)"""
    return np.sqrt(np.average((prediction-prep_x_data(truth))**2))

def bias(truth, prediction):
    """Return the bias between TRUTH (dataseries and PREDICTION (np array)"""
    return np.average(prediction-prep_x_data(truth))

def print_model_diagnostics(model_string, model):
    """Print MODEL diagnostics, labeling them with MODEL_STRING"""
    prediction = model.predict(prep_x_data(causal.SM))
    _bias = bias(causal.ET, prediction)
    _rmse = rmse(causal.ET, prediction)
    print("%s bias: %f" % (model_string, _bias))
    print("%s rmse: %f" % (model_string, _rmse))
    facet = sns.displot(x=causal.ET, y=np.squeeze(prediction))
    ax = facet.ax
    ax.set_title('%s' % model_string)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    lim = [max([xlim[0], ylim[0]]), min([xlim[1], ylim[1]])]
    # ax.plot(lim, lim)
    return (_bias, _rmse)

(bias_naive, rmse_naive) = print_model_diagnostics('Confounding + sampling + model specification', naive_model)
(bias_sample, rmse_sample) = print_model_diagnostics('Sampling + model specification', sample_limited_model)
(bias_sample2, rmse_sample2) = print_model_diagnostics('Sampling2 + model specification', sample_limited_model2)
(bias_confounding, rmse_confounding) =\
    print_model_diagnostics('(Linear) model specification', decorrelated_model)

facet = sns.relplot(data=causal, x='SM', y='ET')
ax = facet.ax
xlim = np.array(ax.get_xlim()).reshape(-1, 1)
for (legend, model) in zip(["naive", "subsampled", "subsampled2", "decorrelated"],
                           [naive_model, sample_limited_model,
                            sample_limited_model2, decorrelated_model]):
    ax.plot(np.squeeze(xlim), np.squeeze(model.predict(xlim)), label=legend)
plt.legend()



plt.show()
