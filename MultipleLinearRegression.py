import numpy as np
import matplotlib.pyplot as pl
import pandas as pd

def selection_sort(x):
    for i in range(len(x)):
        swap = i + np.argmin(x[i:])
        (x[i], x[swap]) = (x[swap], x[i])
    return x

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
le_X = LabelEncoder()
X[:, 3] = le_X.fit_transform(X[:, 3])
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
# print(X)

#Avoiding the Dummy variable trap
X = X[:, 1:] #removing column 0 from the X matrix

# Splitting the dataset into Training and Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# X_train[:, 3:] = sc_X.fit_transform(X_train[:, 3:])
# X_test[:, 3:] = sc_X.transform(X_test[:, 3:])
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
print(X_train)
print(X_test)"""

#Fitting Multiple Linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the test set results
y_pred = regressor.predict(X_test)

#Backward elimination
import statsmodels.api as sm
X = np.append(arr= np.ones((50, 1)).astype(int), values=X.astype(int), axis=1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog= y, exog= X_opt).fit()
p_values = regressor_OLS.summary2().tables[1]['P>|t|'] # a tablazatbol itt eloszor kivesszuk a P ertekeket
print(5.0e-2 < p_values[2]) # az 5.0e-2 felel meg a 0.05-nek
# print(p_values)
sorted_P = (sorted(p_values, key = lambda x:float(x), reverse= True)) # ezzel forditott sorrendbe rendezve egy vektorban megkapjuk a P ertekeket
print(sorted_P[0]) # a vektor 0. eleme lesz a legnagyobb
# print(sorted(p_values))
print(regressor_OLS.summary()) # ez a teljes tablazatot irja ki
