#Importing the libraries
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into Training and Test sets
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
y = y.reshape(-1, 1)
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
y = np.ravel(y)

#Fitting the SVR regression model
from sklearn.svm import SVR
regressor = SVR(kernel= 'rbf')
regressor.fit(X, y)


#Visualizing the SVR regression results
pl.scatter(X, y, color= 'red')
pl.plot(X, regressor.predict(X), color= 'blue')
pl.title('SVR')
pl.xlabel('Position level')
pl.ylabel('Salary')
pl.show()

#Predicting new results with Linear regression
# pred_y = [[None] * 1] * 1
# pred_y[0][0] = 6.5
pred_y = sc_X.transform(np.array([[6.5]]))
#predict the result here e.g.
print(sc_y.inverse_transform(regressor.predict(pred_y)))


X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
pl.scatter(X, y, color= 'red')
pl.plot(X_grid, regressor.predict(X_grid), color= 'blue')
pl.title('Truth or bluff (Polynomial Regression')
pl.xlabel('Position level')
pl.ylabel('Salary')
pl.show()
