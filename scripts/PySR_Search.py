import random as rdm
import os
import numpy as np
import pandas as pd
from pysr import PySRRegressor
from itertools import permutations
from sympy import *
import sympy
import json
from datetime import datetime, timedelta
from sympy.core.rules import Transform

# VERSION 1.9

_kwargs = {
    "version": 19,
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
    "check_if_loss_zero":False,
    "use_best_score": True,
    "penalize_sample_num": False,
    "equation_tracking": False,
    "loss_sample_num": 100000,
    "pysr_params": dict(populations=20)
    }  

# TODO: 
#    - add feynman import
#    - update prediction to jax https://astroautomata.com/PySR/api/#pysr.sr.PySRRegressor.jax

def _make_args(X, variable_num, f):
    X_df = pd.DataFrame(X, columns = [f"x{i}" for i in range(variable_num)])
    args = [X_df[f'{i}'] for i in f.free_symbols]
    return X_df, args
    

def _get_loss(model, x_diff, true_func, args, use_best_score):
    loss_df = model.equations_.copy()
    best_score_index = _best_score_index(model, use_best_score)
    for k in range(len(model.equations_))[best_score_index:]:
        predictions = model.predict(x_diff, index=k).reshape(-1,1)
        real_y = np.array(true_func(*args)).reshape(-1,1)
        loss_df["loss"][k] = (np.square(predictions - real_y)).mean(axis=0)
    return loss_df


def _transform_eq(eq):
    try:
        eq = simplify(eq)
    except:
        pass
    if isinstance(eq, (Float, Integer)):
        return None
    eq = eq.xreplace(Transform(lambda x: 1, lambda x: isinstance(x, (Float)) and x>=0))
    eq = eq.xreplace(Transform(lambda x: -1, lambda x: isinstance(x, (Float)) and x<0))
    return eq


def _check_equal(Expr1,Expr2):
    if Expr1==None or Expr2==None:
        return(False)
    if Expr1.free_symbols!=Expr2.free_symbols:
        return(False)
    vars = Expr1.free_symbols
    your_values=np.random.random(len(vars))
    Expr1_num=Expr1
    Expr2_num=Expr2
    for symbol,number in zip(vars, your_values):
        Expr1_num=Expr1_num.subs(symbol, sympy.Float(number))
        Expr2_num=Expr2_num.subs(symbol, sympy.Float(number))
    Expr1_num=float(Expr2_num)
    Expr2_num=float(Expr2_num)
    if not np.allclose(Expr1_num,Expr2_num):
        return(False)
    try:
        if (Expr1.equals(Expr2)):
            return(True)
    except ValueError:
        return(False)
    except Exception as e:
        print(e)
        return False
    else:
        return(False)
    

def _best_score_index(model, use_best_score):
    best_score_index = model.equations_.score.idxmax()
    if np.abs((len(model.equations_)-1) - best_score_index) < 3:
        best_score_index = ((len(model.equations_)-1) - 3) if (len(model.equations_)-1) >= 3 else 0
    if not use_best_score:
        best_score_index = 0
    return best_score_index
    
def _make_predictions(model, X, use_best_score):
    # calc all predictions for equations with good loss (or since best scoring function)
    best_score_index = _best_score_index(model, use_best_score)
    for idx, k in enumerate(range(len(model.equations_))[best_score_index:]):
        if idx == 0:
            predictions = model.predict(X, index=k).reshape(-1,1)
            continue
        predictions = np.append(predictions, model.predict(X, index=k).reshape(-1,1), axis=1)
    return predictions, best_score_index


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


def check_convergence(model, prev_loss, step, iter_below_tol, loss_iter_below_tol, check_if_loss_zero, loss_df, min_num_iter=25, abs_loss_tol=0.01, rel_loss_tol=0.1):
    """Checks if training for a model has converged."""
    has_converged = False
    #loss = loss_df.loss.mean()
    loss = loss_df.loss.min()
    # check if loss hit zero (true function has been found)
    loss_list = np.array(loss_df.loss.tolist())
    if check_if_loss_zero:
        if True in np.isclose(loss_list, np.zeros(shape=loss_list.shape), atol=1e-6):
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

            
def _sample(confusion, predictions, step_size, test_set, samples, X):
    if step_size > 1 and predictions.shape[1] >= 2:
        sample = _sample_from_distribution(Y=confusion, X=test_set, sample_size=step_size)
        samples = samples.append(sample, ignore_index = True)
        test_set = test_set.drop(sample.index).reset_index(drop=True)
        X = np.delete(X, sample.index, axis=0)
    # find max diff and map it on X to get x value of max diff
    elif predictions.shape[1] >= 2: # If there are more than 1 equation predicted
        max_diff_x = X[np.argmax(confusion)]
        # add new sample points to max diff area
        sample = test_set.iloc[(test_set.loc[:, test_set.columns != 'y']-max_diff_x).sum(axis=1).abs().argsort()[:step_size]]
        samples = samples.append(sample, ignore_index = True)
        test_set = test_set.drop(sample.index).reset_index(drop=True)
        X = np.delete(X, sample.index, axis=0)
    else: # sample randomly
        sample = test_set.sample(n=step_size, random_state=seed)
        samples = samples.append(sample, ignore_index = True)
        test_set = test_set.drop(sample.index).reset_index(drop=True)
        X = np.delete(X, sample.index, axis=0)
    return samples, test_set, X


def _track_equations(prev_equations_df, model, n, penalize_sample_num, use_best_score):
    best_score_index = _best_score_index(model, use_best_score)
    eq_list = []
    for k, eq in enumerate(model.equations_.sympy_format.tolist()[best_score_index:]):
        # simplify eq and remove parameters
        eq = _transform_eq(eq)
        eq_list.append(eq)
    if prev_equations_df.shape[0] == 0:
        # init first df
        prev_equations_df["equation"] = eq_list
        prev_equations_df["occurrence"] = pd.Series(1, index=range(len(eq_list)))
        prev_equations_df["location"] = pd.Series([f"[{n},{x+best_score_index}]" for x in range(len(eq_list))], index=range(len(eq_list)))
        # laborious stuff because of class problems
        prev_equations_df["str"] = prev_equations_df["equation"].astype(str)
        prev_equations_df['index_col'] = prev_equations_df.index
        prev_equations_df_2 = prev_equations_df.groupby(['str'], as_index = False).agg({'occurrence': 'sum', 'location': ','.join, 'index_col':'min'}).reset_index()
        prev_equations_df_2['equation'] = prev_equations_df['equation'].iloc[prev_equations_df_2['index_col'].tolist()].reset_index(drop=True)
        prev_equations_df = prev_equations_df_2.drop(columns=["str", "index_col"])
        return prev_equations_df
    # compare models eqs to previous eqs
    for i, eq_model in enumerate(eq_list):
        # eq_model = sympy.sympify(eq_model)
        if eq_model is None:
            continue
        try:
            k=prev_equations_df["equation"].tolist().index(eq_model)
            prev_equations_df["occurrence"][k] += n if penalize_sample_num else 1
            prev_equations_df["location"][k] += f",[{n},{i+best_score_index}]"
        except:
            prev_equations_df = pd.concat([prev_equations_df, pd.DataFrame({"equation": [eq_model], "occurrence": [1], "location": f"[{n},{i+best_score_index}]"})], ignore_index=True)
                
        # slow code
        """
        Found = False
        for k, eq_df in enumerate(prev_equations_df["equation"].tolist()):
            # eq_df = sympy.sympify(eq_df)
            equal = _check_equal(eq_model,eq_df)
            if equal:
                prev_equations_df["occurrence"][k] += 1 if not penalize_sample_num else n
                prev_equations_df["location"][k] += f",[{n},{i+best_score_index}]"
                Found = True
                break
        if not Found:
            prev_equations_df = pd.concat([prev_equations_df, pd.DataFrame({"equation": [str(eq_model)], "occurrence": [1], "location": f"[{n},{i+best_score_index}]"})], ignore_index=True)
        """
        
        prev_equations_df = prev_equations_df.sort_values(by=['occurrence'], ascending=False).reset_index(drop=True)
    return prev_equations_df
    
    
def _Search(algorithm, eq, seed, N, N_start, N_stop, xstart, xstop, upper_sigma, lower_sigma, niterations, parentdir, binary_operators, unary_operators, denoise, early_stop, loss_iter_below_tol, step_multiplier, check_if_loss_zero, version, use_best_score, penalize_sample_num, equation_tracking, loss_sample_num, pysr_params):
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
    X = np.random.uniform(xstart[0], xstop[0], size=(N, 1))
    if variable_num > 1:
        for i in range(variable_num)[1:]:
            X = np.concatenate((X, np.random.uniform(xstart[i], xstop[i], size=(N, 1))), axis=1)
    X_df, args = _make_args(X, variable_num, f)
    func = lambdify(list(f.free_symbols), f,'numpy')
    sigma = np.random.rand(N) * (upper_sigma - lower_sigma) + lower_sigma
    eps = sigma * np.random.randn(N)
    y = func(*args) + eps

    # First sampling
    test_set = pd.concat([X_df, pd.DataFrame({'y': y})], axis=1)
    samples = test_set.sample(n=N_start, random_state=seed, ignore_index = False)
    test_set = test_set.drop(samples.index).reset_index(drop=True)
    X = np.delete(X, samples.index, axis=0)
    samples = samples.reset_index(drop=True)

    output_path= f'{dir_name}/models.csv'
    samples_output_path=f'{dir_name}/samples.csv'
    equations_output_path=f'{dir_name}/equations.csv'
    loss_output_path=f'{dir_name}/loss.csv'

    # define linspace for evaluating the equations
    iter_below_tol, loss = 0, 0
    has_converged = False
    # loss_sample_num is total number of data points to evaluate equations on (for loss)
    
    x_diff = np.random.uniform(xstart, xstop, size=(loss_sample_num, variable_num)) # dataset for loss evaluation
    X_diff_df, args_diff = _make_args(x_diff, variable_num, f)
    
    td_init_time, time_start = td(time_start_overall)
    time_dict["Initialization time"] = td_init_time
    
    # equation tracker
    prev_equations_df = pd.DataFrame(columns=["equation", "occurrence", "location"])
    
    for _N_start, _N_stop, _steps in _Steps(N_start, N_stop, step_multiplier):
        for i, n in enumerate(np.arange(_N_start, _N_stop, _steps, dtype = int)):
            if i != 0:
                time_start = datetime.now()
                if algorithm == "random":
                    # sample from test set
                    sample = test_set.sample(n=_steps, random_state=seed)
                    samples = samples.append(sample,  ignore_index = True)
                    test_set = test_set.drop(sample.index).reset_index(drop=True)

                if algorithm == "combinatory":
                    time_start = datetime.now()
                    predictions, best_score_index = _make_predictions(model, X, use_best_score=use_best_score)
                    # get diff from every permutation:
                    for num, p in enumerate(_unique_permutations(range(predictions.shape[1]), 2)):
                        if num == 0:
                            differences = np.abs(predictions[:,p[0]]-predictions[:,p[1]]).reshape(-1,1)
                            continue
                        differences = np.append(differences, np.abs(predictions[:,p[0]]-predictions[:,p[1]]).reshape(-1,1), axis=1)
                    differences = differences.sum(axis=1).reshape(-1)
                    samples, test_set, X = _sample(confusion=differences, predictions=predictions, step_size=_steps, test_set=test_set, samples=samples, X=X)

                if algorithm == "std":
                    # Uses standard deviation as confidence score. Uses best equations
                    time_start = datetime.now()
                    predictions, best_score_index = _make_predictions(model, X, use_best_score=use_best_score)
                    std = np.std(predictions, axis=1).reshape(-1)
                    samples, test_set, X = _sample(confusion=std, predictions=predictions, step_size=_steps, test_set=test_set, samples=samples, X=X)

                
                if algorithm == "complexity-std":
                    # Uses standard deviation as confidence score. Uses best equations
                    predictions, best_score_index = _make_predictions(model, X, use_best_score=use_best_score)
                    Sub = np.zeros(predictions.shape)
                    predictions_copy = predictions.copy()
                    for i in range(predictions.shape[0]):
                        Sub[i,:] = np.abs(predictions_copy[i,:] - np.mean(predictions_copy, axis=1)[i])
                    complexity_list = model.equations_.complexity.iloc[best_score_index:].tolist()
                    complexity_std = np.sqrt(np.average(np.square(Sub), axis=1, weights=complexity_list)).reshape(-1)
                    samples, test_set, X = _sample(confusion=complexity_std, predictions=predictions, step_size=_steps, test_set=test_set, samples=samples, X=X)
                    
                if algorithm == "loss-std":
                    # Uses standard deviation as confidence score. Uses best equations
                    time_start = datetime.now()
                    predictions, best_score_index = _make_predictions(model, X, use_best_score=use_best_score)
                    Sub = np.zeros(predictions.shape)
                    predictions_copy = predictions.copy()
                    for i in range(predictions.shape[0]):
                        Sub[i,:] = np.abs(predictions_copy[i,:] - np.mean(predictions_copy, axis=1)[i])
                    loss_list = model.equations_.loss.iloc[best_score_index:].tolist()
                    loss_std = np.sqrt(np.average(np.square(Sub), axis=1, weights=(1/np.array(loss_list)))).reshape(-1)
                    samples, test_set, X = _sample(confusion=loss_std, predictions=predictions, step_size=_steps, test_set=test_set, samples=samples, X=X)

            
            x = samples.iloc[: , :variable_num].to_numpy().reshape(-1,variable_num)
            y = samples.y.to_numpy().reshape(-1,1)
            
            td_model, time_start = td(time_start)
            time_dict["Algorithm time"] = td_model

            # fit model on sample
            model = PySRRegressor(
                niterations=niterations,
                binary_operators=binary_operators,
                unary_operators=unary_operators,
                denoise=denoise,
                model_selection='best',
                temp_equation_file=True,
                tempdir=os.getcwd()+"/tempdir",
                **pysr_params,
                )
            time_start = datetime.now()
            
            model.fit(x, y)
            
            td_model, time_start = td(time_start)
            if i == 0:
                time_dict["First model fit time"] = td_model
            if i == 1:
                time_dict["Second model fit time"] = td_model
            
            
            # track equations between runs
            if equation_tracking:
                prev_equations_df = _track_equations(prev_equations_df, model, n, penalize_sample_num, use_best_score)
            
                td_eq_track, time_start = td(time_start)
                if i == 0:
                    time_dict["First equation tracking"] = td_eq_track
            
                # export equation database
                saved_equations = prev_equations_df.copy()
                saved_equations["sample_size"] = pd.Series(n, index=range(saved_equations.shape[0]))
                saved_equations.to_csv(equations_output_path, mode='a', header=not os.path.exists(equations_output_path))
            
            # export samples
            saved_samples = samples.copy()
            saved_samples["sample_size"] = pd.Series(saved_samples.shape[0], index=range(saved_samples.shape[0]))
            saved_samples.to_csv(samples_output_path, mode='a', header=not os.path.exists(samples_output_path))

            # export models
            saved_model = model.equations_.copy()
            saved_model["sample_size"] = pd.Series(n, index=range(saved_model.shape[0]))
            saved_model.to_csv(output_path, mode='a', header=not os.path.exists(output_path))
            
            # calc loss on true function over large dataset
            loss_df = _get_loss(model=model, x_diff=x_diff, true_func=func, args=args_diff, use_best_score=use_best_score)
            # export updated loss
            saved_loss = loss_df.copy()
            saved_loss["sample_size"] = pd.Series(n, index=range(saved_loss.shape[0]))
            saved_loss.to_csv(loss_output_path, mode='a', header=not os.path.exists(loss_output_path))
            
            if early_stop:
                has_converged, loss, iter_below_tol = check_convergence(model=model, prev_loss=loss, step=n, iter_below_tol=iter_below_tol, loss_iter_below_tol=loss_iter_below_tol, check_if_loss_zero=check_if_loss_zero, loss_df=loss_df)
                if has_converged:
                    break
            if equation_tracking:
                time_dict["Last equation tracking"] = td_eq_track
            # write interim parameter json
            parameter_dict = {"last_n" : int(n), "time": time_dict}
            # write parameters in json file
            with open(f"{dir_name}/parameters.json", "w") as outfile:
                json.dump(parameter_dict, outfile)
            
        if has_converged:
            break
    
    # calculating the execution time
    time_dict["Last model fit time"] = td_model
    
    if equation_tracking:
        time_dict["Last equation tracking"] = td_eq_track
    
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
        "equation_tracking": equation_tracking,
        "early_stop": early_stop,
        "penalize_sample_num": penalize_sample_num,
        "use_best_score": use_best_score,
    }
    # write parameters in json file
    with open(f"{dir_name}/parameters.json", "w") as outfile:
        json.dump(parameter_dict, outfile)
        
    
def Search(**kwargs):
    algorithm = kwargs["algorithm"]
    del kwargs["algorithm"]
    for i, v in kwargs.items():
        _kwargs[i]=v
    _Search(algorithm=algorithm, **_kwargs)