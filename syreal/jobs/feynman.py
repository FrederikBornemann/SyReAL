# CREDIT TO Mile Cranmer. Code heavily inspired by https://github.com/MilesCranmer/PySR/blob/master/pysr/feynman_problems.py
import csv
from pathlib import Path

PKG_DIR = Path(__file__).parents[2]
FEYNMAN_DATASET = PKG_DIR / "feynman.csv"


class Problem:
    """
    Problem API to work with PySR.
    Has attributes: X, y as pysr accepts, form which is a string representing the correct equation and variable_names
    Should be able to call pysr(problem.X, problem.y, var_names=problem.var_names) and have it work
    """

    def __init__(self, X, y, form=None, variable_names=None):
        self.X = X
        self.y = y
        self.form = form
        self.variable_names = variable_names


class FeynmanProblem(Problem):
    """
    Stores the data for the problems from the 100 Feynman Equations on Physics.
    This is the benchmark used in the AI Feynman Paper
    """

    def __init__(self, row):
        """
        row: a row read as a dict from the FeynmanEquations dataset provided in the datasets folder of the repo
        gen: If true the problem will have dp X and y values randomly generated else they will be None
        """
        self.eq_id = row["Filename"]
        self.n_vars = int(row["# variables"])
        super(FeynmanProblem, self).__init__(
            None,
            None,
            form=row["Formula"],
            variable_names=[row[f"v{i + 1}_name"] for i in range(self.n_vars)],
        )
        self.low = [float(row[f"v{i+1}_low"]) for i in range(self.n_vars)]
        self.high = [float(row[f"v{i+1}_high"]) for i in range(self.n_vars)]
        self.boundaries = {name: [float(low), float(high)] for name, low, high in zip(self.variable_names, self.low, self.high)}
        self.dp = row["datapoints"]

    def __str__(self):
        return f"Feynman Equation: {self.eq_id}|Form: {self.form}"

    def __repr__(self):
        return str(self)

def _filter(row, args):
    return [row[arg] == str(args[arg]) for arg in args]

def mk_problems(first=100, Filter=None, data_dir=FEYNMAN_DATASET):
    """
    first: the first "first" equations from the dataset will be made into problems
    data_dir: the path pointing to the Feynman Equations csv
    returns: list of FeynmanProblems
    """
    ret = []
    with open(data_dir) as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            if Filter and (False in _filter(row, Filter)):
                continue
            if i > first:
                break
            if row["Filename"] == "":
                continue
            p = FeynmanProblem(row)
            ret.append(p)
    return ret