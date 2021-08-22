import pandas as pd
import numpy as np
path = "data/"

edges = pd.read_csv(path+"edges",header=None, names=['target','source'])
print(edges.head())

year = pd.read_csv(path+"node_year",header=None)
print(year.head())

features = pd.read_csv(path+"node-feat",header=None)
print(features.head())