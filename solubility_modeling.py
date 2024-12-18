from pycaret.regression import setup
import pandas as pd

data = pd.read_csv('data/logS_data.csv')

setup(data, target='LogS')