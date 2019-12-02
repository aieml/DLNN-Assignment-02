import pandas as pd

dataset=pd.read_csv('heart.csv').as_matrix()
data=dataset[:,0:13]
target=dataset[:,13]
