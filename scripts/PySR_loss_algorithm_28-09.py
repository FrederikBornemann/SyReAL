import random as rdm
import os
import numpy as np
import pandas as pd
from pysr import PySRRegressor
from itertools import permutations

# define first sample set
# fit with that
# evaluate all equations over linspace
# get all permuations of the differences of equations (add them?)
# search for maximum in diff curve
# add sample points at the area of maximum

np.random.seed(0)

# ======== PARAMETERS =========
N = 1000 # Big sample set size
N_start = 2 # How many samples in first sample set 
N_stop = 200 # Where to stop fitting
steps = 1 # sample size increment 
xstart = 0 # interval to sample from
xstop = 2
upper_sigma = 0.5 # error scaling
lower_sigma = 0
# =============================

def true_function(X, eps):
    try: 
        X = X[:,0]
    except:
        pass
    return 5 * np.cos(3.5 * X) - 1.3 + eps

X = (xstop-xstart) * np.random.rand(N, 5) + xstart      # produces X from xstart to xstop
sigma = np.random.rand(N) * (upper_sigma - lower_sigma) + lower_sigma
eps = sigma * np.random.randn(N)
y = true_function(X, eps)

test_set = pd.DataFrame({'x': X[:,0], 'y': y})

output_path= f'models_notrandom_{N_stop}_{upper_sigma}.csv'
samples_output_path=f'samples_notrandom_{N_stop}_{upper_sigma}.csv'

rdm.seed(0)

samples = test_set.iloc[rdm.sample(range(test_set.shape[0]), N_start)]

# outputs unique permutations
def unique_permutations(iterable, r=None):
    previous = tuple()
    for p in permutations(sorted(iterable), r):
        if p > previous:
            previous = p
            yield p

# define linspace for evaluating the equations
x_diff = np.linspace(xstart, xstop, num=50).reshape(-1,1)

for i, n in enumerate(np.arange(N_start, N_stop, steps, dtype = int)):
    # sample from test set
    if i != 0:
        # calc all predictions
        for k in range(len(model.equations_)):
            if k == 0:
                predictions = model.predict(x_diff, index=k).reshape(-1,1)
            predictions = np.append(predictions, model.predict(x_diff, index=k).reshape(-1,1), axis=1)
        # get diff from every permutation:
        for idx, p in enumerate(unique_permutations(range(predictions.shape[1]), 2)):
            if idx == 0:
                differences = np.abs(predictions[:,p[0]]-predictions[:,p[1]]).reshape(-1,1)
            differences = np.append(differences, np.abs(predictions[:,p[0]]-predictions[:,p[1]]).reshape(-1,1), axis=1)
        # find max diff and map it on x_diff to get x value of max diff
        max_diff_x = x_diff[np.argmax(differences.sum(axis=1))]
        # add new sample points to max diff area
        offset = 0
        while True:   # skip to next sample if it is already in sample set
            if test_set.iloc[(test_set['x']-max_diff_x).abs().argsort()[offset:offset+steps]].x.item() not in samples.x.tolist():
                samples = samples.append(test_set.iloc[(test_set['x']-max_diff_x).abs().argsort()[offset:offset+steps]], ignore_index = True)
                break
            offset += 1
            
    x = samples.x.to_numpy().reshape(-1,1)
    y = samples.y.to_numpy().reshape(-1,1)

    # fit model on sample
    model = PySRRegressor(
        niterations=20,
        populations=20,
        binary_operators=["plus", "mult"],
        unary_operators=["cos", "exp", "square"],
        denoise=True,
        model_selection='best',
        select_k_features=1
        )
    model.fit(x, y)

    # export samples
    saved_samples = samples.copy()
    saved_samples["sample_size"] = pd.Series(saved_samples.shape[0], index=range(saved_samples.shape[0]))
    saved_samples.to_csv(samples_output_path, mode='a', header=not os.path.exists(output_path))
    
    # export models
    model.equations_["sample_size"] = pd.Series(n, index=range(model.equations_.score.shape[0]))
    model.equations_.to_csv(output_path, mode='a', header=not os.path.exists(output_path))