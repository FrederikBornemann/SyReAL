import random as rdm
import os
import numpy as np
import pandas as pd
from pysr import PySRRegressor
from itertools import permutations
from sympy import *
import json
from datetime import datetime, timedelta

# VERSION 1.7

_kwargs = {
    "version": 17,
    "eq":"5 * cos(3.5 * x_0) - 1.3",
    "seed":2,
    "N":10000,
    "N_start":2,
    "N_stop":50,
    "xstart":[0.0],
    "xstop":[2.0],
    "upper_sigma":0.5,
    "lower_sigma":0.0,
    "niterations":60,
    "parentdir":os.getcwd(),
    "binary_operators":["plus", "mult"],
    "unary_operators":["cos", "exp", "square"],
    "denoise":False,
    "early_stop":True,
    "loss_iter_below_tol":5,
    "step_multiplier":{0:1},
    "check_if_loss_zero":False
    }  

# TODO: 
#    - add feynman import
#    - when only one equation after best score selection -> take top n EQs

def _sample_from_distribution(Y, X, sample_size):
    # normalize Y values to 1
    pY = np.array(Y) / np.sum(np.array(Y))
    return X.iloc[np.random.choice(X.shape[0], size=sample_size, p=pY, replace=False)]


def td(time_start):
        return str(timedelta(seconds=round((datetime.now() - time_start).total_seconds()))), datetime.now()
    
    
def _Steps(N_start, N_stop, step_multiplier):
    steps_list = list(step_multiplier.values())
    N_start_list = list(step_multiplier)
    for i in range(len(N_start_list)):
        if i>0 and np.abs(N_start_list[i]-N_start_list[i-1])%steps_list[i-1] != 0:
            N_start_list[i]=N_start_list[i-1]+((np.abs(N_start_list[i]-N_start_list[i-1])//steps_list[i-1])+1)*(steps_list[i-1])
    N_start_list[0] = N_start
    N_stop_list = list(step_multiplier)[1:]
    N_stop_list.append(N_stop)
    return zip(N_start_list, N_stop_list, steps_list)


def _float_to_str(f):
    return str(f).replace(".",",").replace("/","÷")


def check_convergence(model, prev_loss, step, iter_below_tol, loss_iter_below_tol, x_diff, true_func, args, check_if_loss_zero, best_score=True, min_num_iter=25, abs_loss_tol=0.01, rel_loss_tol=0.1):
    """Checks if training for a model has converged."""
    has_converged = False
    loss_list = []
    # calc loss
    best_score_index = model.equations_.score.idxmax() if best_score else 0
    # if less than n best equations => evaluate more
    if np.abs((len(model.equations_)-1) - best_score_index) < 3:
        best_score_index = ((len(model.equations_)-1) - 3) if (len(model.equations_)-1) >= 3 else 0
    for k in range(len(model.equations_))[best_score_index:]:
        predictions = model.predict(x_diff, index=k).reshape(-1,1)
        real_y = np.array(true_func(*args)).reshape(-1,1)
        loss_list.append((np.square(predictions - real_y)).mean(axis=0))
    loss_list = np.array(loss_list)
    loss = loss_list.mean()
    # check if loss hit zero (true function has been found)
    if check_if_loss_zero:
        if True in np.isclose(loss_list, np.zeros(shape=loss_list.shape), atol=1e-5):
            has_converged = True
    else:
        # Check if we have reached the desired loss tolerance.
        loss_diff = abs(prev_loss - loss)
        if loss_diff < abs_loss_tol or abs(
            loss_diff / prev_loss) < rel_loss_tol:
            iter_below_tol += 1
        else:
            iter_below_tol = 0
        if iter_below_tol >= loss_iter_below_tol:
            # print('Loss value converged.')
            has_converged = True
        # Make sure that irrespective of the stop criteria, the minimum required
        # number of iterations is achieved.
        if step < min_num_iter:
            has_converged = False
    return has_converged, loss, iter_below_tol 


def _uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1
    while os.path.exists(path):
        path = filename + "(" + str(counter) + ")" + extension
        counter += 1
    return path


# outputs unique permutations
def _unique_permutations(iterable, r=None):
    previous = tuple()
    for p in permutations(sorted(iterable), r):
        if p > previous:
            previous = p
            yield p  
    
    
def Search(algorithm, eq, seed, N, N_start, N_stop, xstart, xstop, upper_sigma, lower_sigma, niterations, parentdir, binary_operators, unary_operators, denoise, early_stop, loss_iter_below_tol, step_multiplier, check_if_loss_zero, version):
    '''
    Takes true equation (eq), generates N data points and searches incrementily over these using the targeted algorithm until N_stop is reached.
    
        Parameters:
                eq (str): True equation in sympy format. Default = "5 * cos(3.5 * x_0) - 1.3".
                seed (int): Seed for random generation. Default = 2.
                N (int): Total number of datapoints generated (density). Default = 1000.
                N_start: How many random datapoints the search starts. Default = 2.
                N_stop (int): At how many datapoints the iteration stops. Default = 50.
                steps (int): How many datapoints are added to the dataset each iteration. Default = 1.
                xstart (float): x starting point. Default = 0.
                xstop (float): x ending point. Default = 2.
                upper_sigma (float): Datapoint max noise scaling. Default = 0.5.
                lower_sigma (float): Datapoint min noise scaling. Default = 0.
                niterations (int): PySR arg -> Number of iterations the search does. Default = 60.
                parentdir (str): Path to the directory for the output. Default = current dir
                binary_operators (list(str)): Binary operators for the model to search for. Default = ["plus", "mult"].
                unary_operators (list(str)): Unary operators for the model to search for. Default = ["cos", "exp", "square"].
                denoise (boolean): If the denoise preprocessing should be used. Default = False.
        Returns: 
                models.csv
                samples.csv
    '''
    # get current time for calculating the execution time
    time_dict = dict()
    time_start_overall = datetime.now()
    
    np.random.seed(seed)
    rdm.seed(seed)

    # Make unique directory
    dir_name = _uniquify(f"{parentdir}/{algorithm}_v{version}_Sigma{_float_to_str(upper_sigma)}_Num{N_stop}_seed{seed}_{_float_to_str(eq)}")
    os.makedirs(dir_name)

    # Parse true equation
    f = sympify(str(eq))
    # evaluate data points
    for i in range(len(f.free_symbols)):
        globals()[f'x{i}'] = symbols(f'x{i}')
    variable_num = np.array([int(str(x).replace("x","")) for x in list(f.free_symbols)]).max()+1
    X = np.random.uniform(xstart[0], xstop[0], size=(N//variable_num, 1))
    if variable_num > 1:
        for i in range(variable_num)[1:]:
            X = np.concatenate((X, np.random.uniform(xstart[i], xstop[i], size=(N//variable_num, 1))), axis=1)
    X_df = pd.DataFrame(X, columns = [f"x{i}" for i in range(variable_num)])
    func = lambdify(list(f.free_symbols), f,'numpy')
    sigma = np.random.rand(N//variable_num) * (upper_sigma - lower_sigma) + lower_sigma
    eps = sigma * np.random.randn(N//variable_num)
    args = [X_df[f'{i}'] for i in f.free_symbols]
    y = func(*args) + eps

    # First sampling
    test_set = pd.concat([X_df, pd.DataFrame({'y': y})], axis=1)
    samples = test_set.sample(n=N_start, random_state=seed, ignore_index = False)
    test_set = test_set.drop(samples.index).reset_index()

    output_path= f'{dir_name}/models.csv'
    samples_output_path=f'{dir_name}/samples.csv'

    # define linspace for evaluating the equations
    iter_below_tol, loss = 0, 0
    has_converged = False
    sample_num = N   # total number of data points to evaluate equations on
    x_diff = X #np.random.uniform(xstart, xstop, size=(sample_num//variable_num, variable_num))
    x_diff = np.delete(x_diff, samples.index, axis=0)
    samples = samples.reset_index(drop=True)
    
    td_init_time, time_start = td(time_start_overall)
    time_dict["Initialization time"] = td_init_time
    
    for _N_start, _N_stop, _steps in _Steps(N_start, N_stop, step_multiplier):
        for i, n in enumerate(np.arange(_N_start, _N_stop, _steps, dtype = int)):

            if algorithm == "random":
                # sample from test set
                sample = test_set.sample(n=_steps, random_state=seed)
                samples = samples.append(sample,  ignore_index = True)
                test_set = test_set.drop(sample.index).reset_index(drop=True)

            if algorithm == "targeted":
                # sample from test set
                if i != 0: # not on first iteration (first sample randomly)
                    time_start = datetime.now()
                    # calc all predictions for equations with good loss (or since best scoring function)
                    best_score_index = model.equations_.score.idxmax()
                    # if less than n best equations => evaluate more
                    if np.abs((len(model.equations_)-1) - best_score_index) < 3:
                        best_score_index = ((len(model.equations_)-1) - 3) if (len(model.equations_)-1) >= 3 else 0
                    for idx, k in enumerate(range(len(model.equations_))[best_score_index:]):
                        if idx == 0:
                            predictions = model.predict(x_diff, index=k).reshape(-1,1)
                            continue
                        predictions = np.append(predictions, model.predict(x_diff, index=k).reshape(-1,1), axis=1)
                    # get diff from every permutation:
                    for num, p in enumerate(_unique_permutations(range(predictions.shape[1]), 2)):
                        if num == 0:
                            differences = np.abs(predictions[:,p[0]]-predictions[:,p[1]]).reshape(-1,1)
                            continue
                        differences = np.append(differences, np.abs(predictions[:,p[0]]-predictions[:,p[1]]).reshape(-1,1), axis=1)
                    differences = differences.sum(axis=1).reshape(-1)
                    if _steps > 1:
                        sample = _sample_from_distribution(Y=differences, X=test_set, sample_size=_steps)
                        samples = samples.append(sample, ignore_index = True)
                        test_set = test_set.drop(sample.index).reset_index(drop=True)
                        x_diff = np.delete(x_diff, sample.index, axis=0)
                    # find max diff and map it on x_diff to get x value of max diff
                    elif predictions.shape[1] >= 2: # If there are more than 1 equation predicted
                        max_diff_x = x_diff[np.argmax(differences)]
                        # add new sample points to max diff area
                        sample = test_set.iloc[(test_set.loc[:, test_set.columns != 'y']-max_diff_x).sum(axis=1).abs().argsort()[:_steps]]
                        samples = samples.append(sample, ignore_index = True)
                        test_set = test_set.drop(sample.index).reset_index(drop=True)
                        x_diff = np.delete(x_diff, sample.index, axis=0)
                    else: # sample randomly
                        sample = test_set.sample(n=_steps, random_state=seed)
                        samples = samples.append(sample, ignore_index = True)
                        test_set = test_set.drop(sample.index).reset_index(drop=True)
                        x_diff = np.delete(x_diff, sample.index, axis=0)
                        
                    td_model, time_start = td(time_start)
                    time_dict["Algorithm time"] = td_model
            
            if algorithm == "active#2":
                # Uses standard deviation as confidence score. Uses best equations
                if i != 0: # not on first iteration (first sample randomly)
                    time_start = datetime.now()
                    # calc all predictions for equations with good loss (or since best scoring function)
                    best_score_index = model.equations_.score.idxmax()
                    if np.abs((len(model.equations_)-1) - best_score_index) < 3:
                        best_score_index = ((len(model.equations_)-1) - 3) if (len(model.equations_)-1) >= 3 else 0
                    for idx, k in enumerate(range(len(model.equations_))[best_score_index:]):
                        if idx == 0:
                            predictions = model.predict(x_diff, index=k).reshape(-1,1)
                            continue
                        predictions = np.append(predictions, model.predict(x_diff, index=k).reshape(-1,1), axis=1)
                    # get mean and std of predictions in every x
                    mean = np.mean(predictions, axis=1).reshape(-1)
                    std = np.std(predictions, axis=1).reshape(-1)
                    if _steps > 1:
                        sample = _sample_from_distribution(Y=std, X=test_set, sample_size=_steps)
                        samples = samples.append(sample, ignore_index = True)
                        test_set = test_set.drop(sample.index).reset_index(drop=True)
                        x_diff = np.delete(x_diff, sample.index, axis=0)
                    # find max std and map it on x_diff to get x value of max diff
                    if predictions.shape[1] >= 2: # If there are more than 1 equation predicted
                        max_diff_x = x_diff[np.argmax(std)]
                        # add new sample points to max diff area
                        sample = test_set.iloc[(test_set.loc[:, test_set.columns != 'y']-max_diff_x).sum(axis=1).abs().argsort()[:_steps]]
                        samples = samples.append(sample,  ignore_index = True)
                        test_set = test_set.drop(sample.index).reset_index(drop=True)
                        x_diff = np.delete(x_diff, sample.index, axis=0)
                    else: # sample randomly
                        sample = test_set.sample(n=_steps, random_state=seed)
                        samples = samples.append(sample,  ignore_index = True)
                        test_set = test_set.drop(sample.index).reset_index(drop=True)
                        x_diff = np.delete(x_diff, sample.index, axis=0)
                        
                    td_model, time_start = td(time_start)
                    time_dict["Algorithm time"] = td_model
            
            x = samples.iloc[: , :variable_num].to_numpy().reshape(-1,variable_num)
            y = samples.y.to_numpy().reshape(-1,1)

            # fit model on sample
            model = PySRRegressor(
                niterations=niterations,
                populations=20,
                binary_operators=binary_operators,
                unary_operators=unary_operators,
                denoise=denoise,
                model_selection='best',
                temp_equation_file=True,
                tempdir=os.getcwd()+"/tempdir"
                )
            time_start = datetime.now()
            
            model.fit(x, y)
            
            td_model, time_start = td(time_start)
            if i == 0:
                time_dict["First model fit time"] = td_model
            if i == 1:
                time_dict["Second model fit time"] = td_model
            
            # export samples
            saved_samples = samples.copy()
            saved_samples["sample_size"] = pd.Series(saved_samples.shape[0], index=range(saved_samples.shape[0]))
            saved_samples.to_csv(samples_output_path, mode='a', header=not os.path.exists(samples_output_path))

            # export models
            model.equations_["sample_size"] = pd.Series(n, index=range(model.equations_.score.shape[0]))
            model.equations_.to_csv(output_path, mode='a', header=not os.path.exists(output_path))

            if early_stop:
                has_converged, loss, iter_below_tol = check_convergence(model=model, prev_loss=loss, step=n, iter_below_tol=iter_below_tol, best_score=True, x_diff=X, true_func=func, args=args, loss_iter_below_tol=loss_iter_below_tol, check_if_loss_zero=check_if_loss_zero)
                if has_converged:
                    break
        if has_converged:
            break

    # calculating the execution time
    time_dict["Last model fit time"] = td_model
    
    td_exec_time, time_start = td(time_start_overall)
    time_dict["Execution time"] = td_exec_time
    
    # collect all parameters
    parameter_dict = {
        "algorithm": algorithm,
        "version": version,
        "time": time_dict,
        "converged": has_converged,
        "last_n" : int(n),
        "equation": eq,
        "seed": seed,
        "N_start": N_start,
        "N_stop": N_stop,
        "steps": step_multiplier,
        "xstart": xstart,
        "xstop": xstop,
        "upper_sigma": upper_sigma,
        "lower_sigma": lower_sigma,
        "niterations": niterations,
        "binary_operators": binary_operators,
        "unary_operators": unary_operators,
        "denoise": denoise,
        "loss_iter_below_tol": loss_iter_below_tol,
    }
    # write parameters in json file
    with open(f"{dir_name}/parameters.json", "w") as outfile:
        json.dump(parameter_dict, outfile)
        
        
def randomSearch(**kwargs):
    for i, v in kwargs.items():
        _kwargs[i]=v
    Search(algorithm="random", **_kwargs)   
    
    
def targetedSearch(**kwargs):
    for i, v in kwargs.items():
        _kwargs[i]=v
    Search(algorithm="targeted", **_kwargs)
    
def targeted2Search(**kwargs):
    for i, v in kwargs.items():
        _kwargs[i]=v
    Search(algorithm="active#2", **_kwargs)