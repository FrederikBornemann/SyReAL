import pandas as pd
import numpy as np
from pathlib import Path

# Set the path to the parent directory of this file
SYREAL_PATH = Path(__file__).resolve().parents[1]
print(SYREAL_PATH)

def import_summary_df():
    """Import the summary dataframe from the data folder."""
    summary_df = pd.read_csv(SYREAL_PATH / 'data/summary_df.csv')
    return summary_df

def str_to_list(string):
    """Convert a string to a list of floats. Replace 'nan' with np.nan. Return a numpy array."""
    string = str(string).replace('[', '').replace(']', '').replace('\n',' ').split(' ')
    string = [x for x in string if x != '']
    string = [float(i) for i in string]
    # replace nan with np.nan
    string = [np.nan if x == 'nan' else x for x in string]
    # convert to numpy array
    string = np.array(string)
    return string

def clean_summary_df(summary_df):
    """Clean the summary dataframe. Convert all columns to lists. Drop the first column. Rename the column "true-confusion" to "true-mod"."""
    # drop the first column
    # summary_df = summary_df.drop(columns=['Unnamed: 0'])
    # rename the column "true-confusion" to "true-mod"
    summary_df.rename(columns={'true-confusion': 'true-mod'}, inplace=True)
    # convert all columns to lists
    for col in summary_df.columns[:-1]:
        summary_df[col] = summary_df[col].apply(str_to_list)
    return summary_df
    
def equation_complexity(equation_string):
    """Calculate the complexity of a given equation string."""
    import sympy
    try:
        # Parse the equation string into a symbolic expression
        expression = sympy.sympify(equation_string)

        # Count the number of operations in the expression
        complexity = sympy.count_ops(expression)

        # Count the number of variables and constants
        num_variables = len(expression.free_symbols)
        num_constants = len(expression.atoms(sympy.Number))

        # Add the counts of variables and constants to the complexity
        total_complexity = complexity + num_variables + num_constants

        return total_complexity
    except (sympy.SympifyError, ValueError):
        return None
    

class Summary():
    """A class to represent the summary dataframe."""
    def __init__(self):
        self.df = import_summary_df()
        self.df = clean_summary_df(self.df)
        self.backup_df = self.df.copy()
        #self.df = self.df.set_index('model')

    def filter(self, threshold_value=50):
        """Filter the dataframe to only include rows where the length of the lists is greater than the threshold value"""
        # make a new dataframe with the length of each list
        summary_df_len = self.df.copy()
        for col in summary_df_len.columns[:-2]:
            summary_df_len[col] = summary_df_len[col].apply(len)

        # Define the columns to check
        columns_to_check = ['random', 'combinatory','std','complexity-std','loss-std','true-mod']	

        # Filter and print rows where at least one element in the specified columns is less than the threshold
        filtered_rows = summary_df_len[(summary_df_len[columns_to_check] < threshold_value).any(axis=1)]
        self.df = self.df.drop(filtered_rows.index)

    def reset(self):
        """Reset the dataframe to the original state."""
        self.df = self.backup_df.copy()

    def equations_of_complexity(self, complexity, geq=False, leq=False):
        """Return a list of equations with the specified complexity"""
        if geq and leq:
            return self.df[self.df['complexity'] >= complexity][self.df['complexity'] <= complexity].index.tolist()
        elif geq:
            return self.df[self.df['complexity'] >= complexity].index.tolist()
        elif leq:
            return self.df[self.df['complexity'] <= complexity].index.tolist()
        else:
            return self.df[self.df['complexity'] == complexity].index.tolist()
    
    

