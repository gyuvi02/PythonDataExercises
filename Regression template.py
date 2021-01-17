import numpy as np
import matplotlib.pyplot as pl
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into Training and Test sets
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# X_train[:, 3:] = sc_X.fit_transform(X_train[:, 3:])
# X_test[:, 3:] = sc_X.transform(X_test[:, 3:])
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
print(X_train)
print(X_test)"""

#Fitting Linear regression - to compare with polynimial
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Fitting Multiple Linear regression to the training set
"""from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)"""

#Fitting Polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poli_reg = PolynomialFeatures(degree= 4)
X_poly = poli_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)
