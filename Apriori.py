#Importing the libraries
import matplotlib
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
transactions = []
for i in range(0, 7501):
   transactions.append([str(dataset.values[i,j]) for j in range(0,20)])

# Training apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support= 3*7/7500, min_confidence= 0.2, min_lift= 3)

# Visualizing the results
results = list(rules)
