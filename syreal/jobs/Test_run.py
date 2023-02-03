import sys
import os
from PySR_Search import Search
#from PySR_Graphics import lossPlotSingle, lossComparisonPlot, Animation

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


eq = "exp(-((theta-theta1)/sigma)**2/2)/(sqrt(2*3.1415)*sigma)"
boundaries = {"sigma":[1.0,3.0],"theta":[1.0,3.0],"theta1":[1.0,3.0]}
seed = 3
parentdir=f"{os.getcwd()}/Output/Tests/TEST_feynman_algos_{seed}"
Plots = False
    
args = dict(
    eq=eq,
    upper_sigma=0.0, 
    niterations=30, 
    boundaries=boundaries,
    parentdir=parentdir, 
    unary_operators=["neg","square","cube","exp","sqrt","sin","cos","tanh"], 
    binary_operators=["plus","sub","mult","div"],
    check_if_loss_zero=True, 
    N_stop=10,
    N_start=5,
    seed=seed,
    equation_tracking=True,
    early_stop=False,
)
#Search(algorithm="random", **args)
Search(algorithm="combinatory", **args)
Search(algorithm="std", **args)
Search(algorithm="complexity_std", **args)
Search(algorithm="loss_std", **args)
Search(algorithm="true_confusion", **args)

if Plots:
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
        
        