import numpy as np
import matplotlib.pyplot as pl
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
# print(X)
# print(y)

# Splitting the dataset into Training and Test sets
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Fitting Linear regression - to compare with polynimial
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Fitting Polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poli_reg = PolynomialFeatures(degree= 4)
X_poly = poli_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

#Visualizing the Linear regression results
pl.scatter(X, y, color= 'red')
pl.plot(X, lin_reg.predict(X), color= 'blue')
pl.title('Truth or bluff (Linear Regression')
pl.xlabel('Position level')
pl.ylabel('Salary')
# pl.show()


#Visualizing the Polynomial regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
pl.scatter(X, y, color= 'red')
pl.plot(X_grid, lin_reg2.predict(poli_reg.fit_transform(X_grid)), color= 'blue')
pl.title('Truth or bluff (Polynomial Regression')
pl.xlabel('Position level')
pl.ylabel('Salary')
# pl.show()

#Predicting new results with Linear regression
A = [[None] * 1] * 1
A[0][0] = 6.5
print(lin_reg.predict(A))


#Predicting new results with Polynomial regression
print(lin_reg2.predict(poli_reg.fit_transform(A)))


