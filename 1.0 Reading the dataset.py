import pandas as pd

dataset=pd.read_csv('heart.csv').values
#this code read the whole dataset and store it in a numpy array
data=dataset[:,0:13]
target=dataset[:,13]
