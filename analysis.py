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

data_dir = "./data"

experiments = ["causal", "decorrelated", "reality"]

def load_experiment(name):
    "Load experiment NAME."
    df = pd.read_csv("%s/%s.csv" % (data_dir, name))
    return df[~np.isnan(df.ET)]

causal = load_experiment("causal")
decorrelated = load_experiment("decorrelated")
reality = load_experiment("reality")
