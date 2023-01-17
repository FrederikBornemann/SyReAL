import sys
import os
from PySR_Search import Search
from PySR_Graphics import lossPlotSingle, lossComparisonPlot, Animation

# Change cwd to 'home' directory
import sys, os
# initial directory
cwd = os.getcwd()
# directory of "Output" folder
d = '/home/bornemaf/'
# trying to insert to directory
try:
    print("Inserting inside-", os.getcwd())
    os.chdir(d)      
except:
    print("Something wrong with specified directory. Exception- ")
    print(sys.exc_info())


def indices(lst, item):
    return [i for i, x in enumerate(lst) if x == item]

#equations = ["exp(-((x0/x1)**2)/2)/(2.5*x1)", "exp(-(((x0-x1)/x2)**2)/2)/(2.5*x2)"]
equations = ["exp(-(((x0-x1)/x2)**2)/2)/(2.5*x2)"]
#xstop = [[3.0,3.0],[3.0,3.0,3.0]]
xstop = [[3.0,3.0,3.0]]
#xstart = [[1.0,1.0],[1.0,1.0,1.0]]
xstart = [[1.0,1.0,1.0]]



for seed in [3,4,5,6]:
    parentdir=f"{os.getcwd()}/Output/Tests/TEST_feynman_algos_{seed}"
    for i, eq in enumerate(equations):
        args = dict(
            eq=eq,
            upper_sigma=0.0, 
            niterations=30, 
            xstop=xstop[i], 
            xstart=xstart[i], 
            parentdir=parentdir, 
            unary_operators=["cos", "exp", "square","sqrt"], 
            binary_operators=["plus", "mult","div"], 
            check_if_loss_zero=True, 
            N_stop=7, 
            seed=seed,
            equation_tracking=True,
            early_stop=False,
        )
        Search(algorithm="random", **args)
        Search(algorithm="combinatory", **args)
        Search(algorithm="std", **args)
        Search(algorithm="complexity_std", **args)
        Search(algorithm="loss_std", **args)
        pass
    # Make plots
    subfolders = [ f.path for f in os.scandir(parentdir) if f.is_dir() ]
    for folder in subfolders:
        if ".ipynb" in folder:
            continue
        try:
            #lossPlotSingle(dirname=folder, best_score=True)
            pass
        except:
            pass

    # Make comparison plots
    folder_eq = [x.split("_")[-1].replace(",",".").replace("÷","/") for x in subfolders]
    for eq in equations:
        dirnames = [subfolders[i] for i in indices(folder_eq, eq)]
        if len(dirnames) >= 2:
            try:
                lossComparisonPlot(dirnames=dirnames, parentdirname=parentdir)
                pass
            except:
                pass
            try:
                #Animation(dirnames=dirnames, parentdirname=parentdir)
                pass
            except:
                pass
        
        