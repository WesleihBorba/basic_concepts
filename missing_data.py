# Goal: Treat data with missing values
import pandas as pd


class MissingData:

    def __init__(self):
        self.data_missing = pd.DataFrame()

    def read_file(self):
        pass



    def drop(self):
        self.data_missing.dropna(inplace=True)

# PAREI AQUI: Missing at Random (MAR)