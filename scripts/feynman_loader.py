import pandas as pd
import os
from read_config import read_config
from os import listdir
from os.path import isfile, join


def load_feynman_csv():
    '''Loads the Feynman Equations CSV from the PySR Repo. Output: pandas.DataFrame'''
    url = 'https://raw.githubusercontent.com/MilesCranmer/PySR/master/datasets/FeynmanEquations.csv'
    return pd.read_csv(url)

def load_datapoints(path=None, files=[], inside=True):
    config = read_config(inside=inside)
    if len(files) == 0:
        raise Exception("No files (equation file names) were specified")
    if path == None:
        path = config["directory"]["feynmanDatapoints"]
    if not os.path.exists(path):
        raise Exception("Path doesn't exist")
        return None
    files_in_dir = [f for f in listdir(path) if isfile(join(path, f))]
    if len(files_in_dir) == 0:
        raise Exception(f"No datapoint files in {path}")
    DF_list = list()
    for file in files:
        if file not in files_in_dir:
            raise Exception(f"{file} was not found in specified directory")
        DF_list.append(pd.read_csv(join(path,file), delim_whitespace=True, header=None))
    return DF_list
        
    
    

    
    