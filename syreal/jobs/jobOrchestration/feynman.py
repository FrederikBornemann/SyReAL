import csv
from constants import FEYNMAN_CSV_FILE

class Feynman():
    def __init__(self, name):
        self.name = name
        self.equation = None
        self.boundaries = {}
        self.datapoints = None
        self.variable_names = []
        self.set_attributes_from_csv()

    def set_attributes_from_csv(self):
        with open(FEYNMAN_CSV_FILE, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0] == self.name:  # Find the row that matches the name
                    self.datapoints = int(row[1])
                    self.equation = str(row[4])
                    names = [str(row[6 + i * 3]) for i in range(int(row[5]))]
                    self.variable_names = names
                    boundaries = {name: [float(row[7 + i * 3]), float(row[8 + i * 3])] for i, name in enumerate(names)}
                    self.boundaries = boundaries
                    break
    
