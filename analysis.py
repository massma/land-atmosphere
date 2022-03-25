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

experiments = ["kelowna-causal", "kelowna-decorrelated", "kelowna-reality"]

def load_experiment(name):
    "Load experiment NAME."
    df = pd.read_csv("%s/%s.csv" % (data_dir, name))
    return df[~np.isnan(df.ET)]

NSAMPLE = 10
causal = load_experiment("kelowna-causal")
decorrelated = load_experiment("kelowna-decorrelated")
reality = load_experiment("kelowna-reality")
subsamples = [decorrelated.sample(n=reality.shape[0], random_state=i) for i in range(NSAMPLE)]

naive_model = LinearRegression()
sample_models = [LinearRegression() for _s in subsamples]
decorrelated_model = LinearRegression()

def prep_x_data(ds):
    """Take a dataseries DS and make it into the form needed by scikit fit."""
    return ds.to_numpy().reshape(-1, 1)


naive_model.fit(X=prep_x_data(reality.SM), y=reality.ET)

for (model, data) in zip(sample_models, subsamples):
    model.fit(X=prep_x_data(data.SM), y=data.ET)

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
bias_rmse_samples = [print_model_diagnostics('Sampling %d + model specification' % i, model)
                     for (i, model) in zip(range(NSAMPLE), sample_models)]
(bias_confounding, rmse_confounding) =\
    print_model_diagnostics('(Linear) model specification', decorrelated_model)

facet = sns.relplot(data=causal, x='SM', y='ET')
ax = facet.ax
xlim = np.array(ax.get_xlim()).reshape(-1, 1)

for (i, model) in zip(range(NSAMPLE), sample_models):
    ax.plot(np.squeeze(xlim), np.squeeze(model.predict(xlim)), label=None,linewidth=0.4)
ax.plot(np.squeeze(xlim), np.squeeze(naive_model.predict(xlim)), label="naive")
ax.plot(np.squeeze(xlim), np.squeeze(decorrelated_model.predict(xlim)),
        label="decorrelated")

plt.legend()
plt.show()
