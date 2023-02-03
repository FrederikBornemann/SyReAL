from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from sympy import *
import json
import matplotlib.gridspec as gridspec
from celluloid import Camera
import os


def _add_func_to_models(parameters, models):
    EQ = parameters["equation"]
    f = sympify(str(EQ))
    for i in range(len(f.free_symbols)):
        globals()[f'x{i}'] = symbols(f'x{i}')
    variable_num = np.array([int(str(x).replace("x","")) for x in list(f.free_symbols)]).max()+1
    for i, eq in enumerate(models.sympy_format):
        _f = sympify(str(eq))
        _func = lambdify(list(f.free_symbols), _f,'numpy')
        models.lambda_format[i] = _func
    return models
    
def _read_parameters(dirname):
    with open(f'{dirname}/parameters.json', 'r') as openfile:
        json_object = json.load(openfile)
    return json_object

def _calc_loss_plot_data(models, sample_num, best_score, parameters):
    
    EQ = parameters["equation"]
    xstart = parameters["xstart"]
    xstop = parameters["xstop"]
    
    # Parse true equation
    f = sympify(str(EQ))
    # evaluate data points
    for i in range(len(f.free_symbols)):
        globals()[f'x{i}'] = symbols(f'x{i}')
    variable_num = np.array([int(str(x).replace("x","")) for x in list(f.free_symbols)]).max()+1
    x_con = np.random.uniform(xstart, xstop, size=(sample_num//variable_num, variable_num))
    X_df = pd.DataFrame(x_con, columns = [f"x{i}" for i in range(variable_num)])
    func = lambdify(list(f.free_symbols), f,'numpy')
    args = [X_df[f'{i}'] for i in f.free_symbols]
    real_y_con = np.array(func(*args)).reshape(-1,1)

    losses = pd.DataFrame({"sample_size": [], "equation": [], "loss": []})

    # loop over all sample_sizes
    for idx, n in enumerate(models.sample_size.unique().tolist()):
        equations = models.loc[models["sample_size"] == n]
        best_score_idx = equations.score.reset_index(drop=True).idxmax() if best_score else 0
        # loop over all equations in the model for sample_size == n
        for eq in equations.lambda_format[best_score_idx:]:
            y_con = np.array(eq(*args)).reshape(-1,1)
            loss = (np.square(y_con - real_y_con)).mean(axis=0)
            df = pd.DataFrame({"sample_size": [n], "equation": [eq], "loss": [loss]})
            losses = losses.append(df, ignore_index = True)


    loss_plot = pd.DataFrame({"sample_size": [], "min_loss": [], "mean_loss": [], "loss_err": []})

    for n in losses.sample_size.unique().tolist():
        loss_list = np.array(losses.loc[losses['sample_size']==n,'loss'].tolist())

        df = pd.DataFrame({"sample_size": [n], "min_loss": [loss_list.min()], "max_loss": [loss_list.max()], "mean_loss": [loss_list.mean()], "loss_err": [loss_list.std()]})
        loss_plot = loss_plot.append(df, ignore_index = True)

    return loss_plot


def lossPlotSingle(dirname, N_limit=None, best_score=False, sample_num=10000):
    '''
    Returns a plot of the evolution of the loss of the model. Including the mean, min, max and std.
    
        Parameters:
                dirname (str): The directory in which the model.csv and parameters.json file are located
                
    '''
    # read parameters and models out of directory
    parameters = _read_parameters(dirname)
    models = pd.read_csv(f'{dirname}/models.csv')
    models = _add_func_to_models(parameters, models)
    
    # calc the loss plot
    loss_plot = _calc_loss_plot_data(models, sample_num, best_score=best_score, parameters=parameters)
    
    # --- PLOT ---
    convergence = np.array(loss_plot.mean_loss.tolist()[-parameters["loss_iter_below_tol"]:]).mean() if parameters["converged"] else np.array(loss_plot.mean_loss.tolist()[-5:]).mean()
    plt.figure(figsize=(13, 10), dpi= 100, facecolor='w', edgecolor='k')

    if N_limit is not None:
        try:
            loss_lim_idx = loss_plot.sample_size.tolist().index(N_limit)
        except:
            loss_lim_idx = len(loss_plot.sample_size.tolist())-1
            print(f"N_limit is too high. Highest sample size is {loss_plot.sample_size.tolist()[-1]}")
    else:
        loss_lim_idx = len(loss_plot.sample_size.tolist())-1
    
    mins = loss_plot.min_loss[:loss_lim_idx]
    maxes = loss_plot.max_loss[:loss_lim_idx]
    means = loss_plot.mean_loss[:loss_lim_idx]
    std = loss_plot.loss_err[:loss_lim_idx]

    plt.errorbar(loss_plot.sample_size.tolist()[:loss_lim_idx], means, std, fmt='ok', lw=3, ecolor="lightblue", label="loss std")
    plt.errorbar(loss_plot.sample_size.tolist()[:loss_lim_idx], means, [means - mins, maxes - means], fmt='.k', ecolor='darkred', lw=1, label="loss mean(black) and min to max loss")

    plt.hlines(convergence, xmin=loss_plot.sample_size.tolist()[0], xmax=loss_plot.sample_size.tolist()[loss_lim_idx], color="red", label=f"loss limit: {round(convergence,3)}")

    # legend modification
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    #handles = [h[0] for h in handles]
    plt.rcParams['font.size'] = '16'
    ax.legend(handles, labels, loc='upper left',numpoints=1,  fontsize=15)

    # text placement
    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height
    
    ax.text(right, top, parameters["equation"], horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
    
    ax.set_ylim([0, means[0]])
    plt.xlabel("Sample size", fontsize=15) 
    plt.ylabel("Loss (MSE)", fontsize=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    #plt.legend()
    plt.title(f"Average Loss(MSE) of the equations per training sample size ({parameters['algorithm']}): Sigma={parameters['upper_sigma']}", fontsize=20)
    fig = plt.gcf()
    plt.draw()
    plt.show()
    filename = f"{dirname}/{parameters['algorithm']}_loss_single.png"
    fig.savefig(filename, dpi=150, bbox_inches='tight')

    
def lossComparisonPlot(dirnames, parentdirname, N_limit=None, best_score=False, sample_num=10000, return_fig=False, axis=None):
    palette = cm.get_cmap('tab10')
    colors =  {"random": "blue", "combinatory": "red", "std": "green", "complexity-std": "purple", "loss-std": "cyan"}
    num = len(dirnames)
    
    fontsize_1 = 15 if not return_fig else 7
    fontsize_2 = 13 if not return_fig else 5
    fontsize_3 = 20 if not return_fig else 18

    plt.figure(figsize=(13, 10), dpi= 100, facecolor='w', edgecolor='k')
    if return_fig:
        plt.sca(axis)
    return_list = []
    loss_max_list = []
    for i, dirname in enumerate(dirnames):
        parameters = _read_parameters(dirname)
        models = pd.read_csv(f'{dirname}/models.csv')
        models = _add_func_to_models(parameters, models)
        #loss = pd.read_csv(f'{dirname}/loss.csv')
        
        loss_plot = _calc_loss_plot_data(models, sample_num, best_score=best_score, parameters=parameters)
        
        convergence = np.array(loss_plot.min_loss.tolist()[-parameters["loss_iter_below_tol"]:]).mean() if parameters["converged"] else np.array(loss_plot.min_loss.tolist()[-5:]).mean()
        
        color = colors[parameters["algorithm"]]
        
        if N_limit is not None:
            try:
                loss_lim_idx = loss_plot.sample_size.tolist().index(N_limit)
            except:
                loss_lim_idx = len(loss_plot.sample_size.tolist())-1
                print(f"N_limit is too high. Highest sample size is {loss_plot.sample_size.tolist()[-1]}")
        else:
            loss_lim_idx = len(loss_plot.sample_size.tolist())-1
        
        # CHANGE FROM MEAN TO MIN!!!
        means = loss_plot.min_loss[:loss_lim_idx]
        loss_max_list.append(means[0])
        
        plt.scatter(loss_plot.sample_size.tolist()[:loss_lim_idx], means, color=color, label=f"({parameters['algorithm']}) min loss", alpha=0.5)
        plt.plot(loss_plot.sample_size.tolist()[:loss_lim_idx], means, color=color, alpha=0.5)
        plt.hlines(convergence, xmin=loss_plot.sample_size.tolist()[0], xmax=loss_plot.sample_size.tolist()[loss_lim_idx], color=color, label=f"({parameters['algorithm']}) min loss limit: {round(convergence,3)}")

        return_list.append([loss_plot, means, convergence, loss_lim_idx, parameters, models])
    

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    # handles = [h[0] for h in handles]
    ax.legend(handles, labels, loc='upper right',numpoints=1, fontsize=fontsize_1)
    
    # text placement
    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height
    
    if not return_fig:
        ax.text(right, top, _read_parameters(dirnames[0])["equation"], horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=fontsize_1)
    
    ax.set_ylim([0, np.array(loss_max_list).max()])
    plt.xlabel("Sample size",fontsize=fontsize_1) 
    plt.ylabel("Loss (MSE)",fontsize=fontsize_1)
    plt.xticks(fontsize=fontsize_2)
    plt.yticks(fontsize=fontsize_2)
    #plt.legend()
    if not return_fig:
        plt.title(f"Comparison of minimal losses: Sigma={_read_parameters(dirnames[0])['upper_sigma']}",fontsize=fontsize_3)
    fig = plt.gcf()
    if not return_fig:
        plt.draw()
        plt.show()
        filename = f"{parentdirname}/{_read_parameters(dirnames[0])['equation'].replace('.',',').replace('/','÷')}_loss_comparison.png"
        fig.savefig(filename, dpi=150, bbox_inches='tight')
    else:
        return_list.append([np.array(loss_max_list), palette])
        return plt.gca(), return_list
    
    


def Animation(dirnames, parentdirname, plot_variable_idx=0):
    fig = plt.figure(figsize=(11, 7), dpi=200)
    camera = Camera(fig)
    G = gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(G[0, :])
    ax2 = plt.subplot(G[1, 0])
    ax3 = plt.subplot(G[1, 1])
    ax2.set(xlabel=f'x{plot_variable_idx}')
    ax3.set(xlabel=f'x{plot_variable_idx}')
    _ax1, return_list = lossComparisonPlot(dirnames=dirnames, parentdirname=parentdirname, best_score=True, axis=ax1, return_fig=True)
    palette = cm.get_cmap('tab10') #return_list[-1][1]
    models = [x[5] for x in return_list[:-1]]
    nstart, nstop = min(models[0]["sample_size"].unique().tolist()),0
    for model in models:
        nstop = max(model["sample_size"].unique().tolist()) if max(model["sample_size"].unique().tolist()) > nstop else nstop
    for n in range(nstart, nstop):
        plt.sca(ax1)
        for i in range(len(return_list)-1):
            loss_plot = return_list[i][0]
            means = return_list[i][1]
            convergence = return_list[i][2]
            loss_lim_idx = return_list[i][3]
            parameters = return_list[i][4]
            color = "blue" if parameters["algorithm"] == "random" else "orange"
            plt.scatter(loss_plot.sample_size.tolist()[:loss_lim_idx], means, color=color, label=f"({parameters['algorithm']}) mean loss", s=10)
            plt.plot(loss_plot.sample_size.tolist()[:loss_lim_idx], means, color=color)
            plt.hlines(convergence, xmin=loss_plot.sample_size.tolist()[0], xmax=loss_plot.sample_size.tolist()[loss_lim_idx], color=color, label=f"({parameters['algorithm']}) loss limit: {round(convergence,3)}")
        ax1.set_ylim([0, return_list[-1][0].max()])
        ax1.text(n+0.2, return_list[-1][0]/2, f"N={n}", verticalalignment='center')
        #ax1.set_yscale('log')
        plt.xlabel("Sample size",fontsize=7) 
        plt.ylabel("Loss (MSE)",fontsize=7)
        plt.xticks(fontsize=5)
        plt.yticks(fontsize=5)
        #plt.legend()
        plt.title(f"Comparison of average losses (random vs targeted)",fontsize=18)
        ax1.vlines(n, 0, return_list[-1][0].max())
        
        # lower plots:
        # Parse true equation
        sample_num=500
        parameters = return_list[0][4]
        EQ = parameters["equation"]
        xstart = parameters["xstart"]
        xstop = parameters["xstop"]
        f = sympify(str(EQ))
        # evaluate data points
        for i in range(len(f.free_symbols)):
            globals()[f'x{i}'] = symbols(f'x{i}')
        variable_num = np.array([int(str(x).replace("x","")) for x in list(f.free_symbols)]).max()+1
        x_con = np.linspace(xstart[plot_variable_idx],xstop[plot_variable_idx],sample_num).reshape(-1,1)
        x_con_plot = np.copy(x_con)
        if variable_num > 1:
            for i in range(variable_num):
                if i == plot_variable_idx:
                    continue
                x_con = np.concatenate((x_con, np.full((sample_num,1), np.array([xstart[i],xstop[i]]).mean()).reshape(-1,1)), axis=1)
        X_df = pd.DataFrame(x_con, columns = [f"x{i}" for i in range(variable_num)])
        func = lambdify(list(f.free_symbols), f,'numpy')
        args = [X_df[f'{i}'] for i in f.free_symbols]
        real_y_con = np.array(func(*args)).reshape(-1,1)
        for i in range(len(dirnames)):
            samples = pd.read_csv(f'{dirnames[i]}/samples.csv')
            ani_samples = samples.copy().drop_duplicates(subset=[f'x{plot_variable_idx}', 'y']).reset_index(drop=True)
            plt.sca(plt.subplot(G[1, i]))

            # plot samples
            plt.scatter(ani_samples.loc[ani_samples["sample_size"]<n][f'x{plot_variable_idx}'], ani_samples.loc[ani_samples["sample_size"]<n].y, color="black", zorder=2)
            plt.scatter(ani_samples.loc[ani_samples["sample_size"]==n][f'x{plot_variable_idx}'], ani_samples.loc[ani_samples["sample_size"]==n].y, color="yellow", edgecolor="black", zorder=3)

            df = models[i].loc[models[i]["sample_size"]==n]
            # get number of equations
            idx = len(df)
            # plot every equation from the model
            palette = cm.get_cmap('hsv', idx)
            eq_list = []
            lines = []

            line, = plt.plot(x_con_plot, real_y_con, linewidth=2, color="black") #label="True function"
            lines.append(line)
            eq_list.append(f"${latex(sympify(parameters['equation']))}$")
            if i == 0:
                bottom, top = plt.ylim()
            else:
                plt.ylim((bottom, top))   # set the ylim to bottom, top

            for j, eq in enumerate(df.lambda_format.tolist()):
                # if best score => bold label
                preamble, endamble = ["", ""]
                if j == df.score.reset_index(drop=True).idxmax():
                    preamble, endamble = ["\\textbf{", "}"]
                
                y_con = np.array(eq(*args)).reshape(-1,1)
                if y_con.shape == (1,1):
                    continue
                
                line, = plt.plot(x_con_plot, y_con, linewidth=1, alpha=0.75, color=palette(j)) #label=f"{preamble}score: {round(df.score.iloc[j],3)}{endamble}"
                lines.append(line)
                
                # round parameters for legend
                try: 
                    eq = sympify(df.sympy_format.reset_index(drop=True)[j])
                    for a in preorder_traversal(eq):
                        if isinstance(a, Float):
                            eq = eq.subs(a, round(a, 2))
                    eq_list.append(f"${latex(eq)}$")
                except:
                    #print(n,i,j)
                    pass
            
            legend1 = plt.legend(lines, eq_list, bbox_to_anchor=(0.5, -0.05), loc='upper center', ncol=2, fancybox=False, prop={'size': 5})
            plt.gca().add_artist(legend1)
            plt.title(f"{return_list[i][4]['algorithm']}", fontsize=10)
            plt.legend(loc=3)
        camera.snap()
    # plt.show()
    filename = f"{parentdirname}/{_read_parameters(dirnames[0])['equation'].replace('.',',').replace('/','÷')}_Animation.gif"
    anim = camera.animate(interval = 400)
    anim.save(filename)
    
    
def equation_bar_chart_animation(dirname):
   
    def top_n_equations(sample_size, equation_tracker, n):
        equations = equation_tracker.iloc[equation_tracker["sample_size"] == sample_size]
        return top_n_eqs, tickdic
    
    
    from matplotlib import animation
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import seaborn as sns
    
    fig = plt.figure(figsize=(8,6))
    axes = fig.add_subplot(1,1,1)
    axes.set_ylim(0, 150)
    plt.style.use("seaborn")
    
    palette = list(reversed(sns.color_palette("seismic", 2).as_hex()))
    
    # import data
    parameters = _read_parameters(dirname)
    equation_tracker = pd.read_csv(f'{dirname}/equations.csv')
    
    N = np.unique(equation_tracker['sample_size'].values)
    pass


def lossComparisonPlot2(dirnames, parentdirname, N_limit=None, best_score=False, sample_num=10000, return_fig=False, axis=None):
    palette = cm.get_cmap('tab10')
    colors =  {"random": "blue", "combinatory": "red", "std": "green", "complexity-std": "purple", "loss-std": "cyan"}
    num = len(dirnames)
    
    fontsize_1 = 15 if not return_fig else 7
    fontsize_2 = 13 if not return_fig else 5
    fontsize_3 = 20 if not return_fig else 18

    plt.figure(figsize=(13, 10), dpi= 100, facecolor='w', edgecolor='k')
    if return_fig:
        plt.sca(axis)
    return_list = []
    loss_max_list = []
    for i, dirname in enumerate(dirnames):
        parameters = _read_parameters(dirname)
        #models = pd.read_csv(f'{dirname}/models.csv')
        #models = _add_func_to_models(parameters, models)
        models = None
        loss = pd.read_csv(f'{dirname}/loss.csv')
        
        #loss_plot = _calc_loss_plot_data(models, sample_num, best_score=best_score, parameters=parameters)
        loss_plot = loss[['loss', 'sample_size']].groupby("sample_size", as_index=False).min()
        convergence = None #np.array(loss_plot.min_loss.tolist()[-parameters["loss_iter_below_tol"]:]).mean() if parameters["converged"] else np.array(loss_plot.min_loss.tolist()[-5:]).mean()
        
        color = colors[parameters["algorithm"]]
        
        if N_limit is not None:
            try:
                loss_lim_idx = loss_plot.sample_size.tolist().index(N_limit)
            except:
                loss_lim_idx = len(loss_plot.sample_size.tolist())-1
                print(f"N_limit is too high. Highest sample size is {loss_plot.sample_size.tolist()[-1]}")
        else:
            loss_lim_idx = len(loss_plot.sample_size.tolist())-1
        
        # CHANGE FROM MEAN TO MIN!!!
        means = loss_plot.loss.tolist()[:loss_lim_idx]
        print(means)
        loss_max_list.append(means[0])
        
        plt.scatter(loss_plot.sample_size.tolist()[:loss_lim_idx], means, color=color, label=f"({parameters['algorithm']}) min loss", alpha=0.5)
        plt.plot(loss_plot.sample_size.tolist()[:loss_lim_idx], means, color=color, alpha=0.5)
        #plt.hlines(convergence, xmin=loss_plot.sample_size.tolist()[0], xmax=loss_plot.sample_size.tolist()[loss_lim_idx], color=color, label=f"({parameters['algorithm']}) min loss limit: {round(convergence,3)}")
        return_list.append([loss_plot, means, convergence, loss_lim_idx, parameters, models])
    

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    # handles = [h[0] for h in handles]
    ax.legend(handles, labels, loc='upper right',numpoints=1, fontsize=fontsize_1)
    
    # text placement
    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height
    
    if not return_fig:
        ax.text(right, top, _read_parameters(dirnames[0])["equation"], horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=fontsize_1)
    
    ax.set_ylim([0, np.array(loss_max_list).max()])
    plt.xlabel("Sample size",fontsize=fontsize_1) 
    plt.ylabel("True loss (MSE)",fontsize=fontsize_1)
    plt.xticks(fontsize=fontsize_2)
    plt.yticks(fontsize=fontsize_2)
    #plt.legend()
    if not return_fig:
        plt.title(f"Comparison of minimal losses: Sigma={_read_parameters(dirnames[0])['upper_sigma']}",fontsize=fontsize_3)
    fig = plt.gcf()
    if not return_fig:
        plt.draw()
        plt.show()
        filename = f"{parentdirname}/{_read_parameters(dirnames[0])['equation'].replace('.',',').replace('/','÷')}_loss_comparison2.png"
        fig.savefig(filename, dpi=150, bbox_inches='tight')
    else:
        return_list.append([np.array(loss_max_list), palette])
        return plt.gca(), return_list