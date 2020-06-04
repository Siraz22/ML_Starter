#Part 1 : Data preprocessing Template

#Importing libraries
import numpy as np #mathematical tools
import matplotlib.pyplot as plt
import pandas as pd #import dataset lib

#importing the datasets
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values

#Dummy Variables
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy Variable Trap explicitly by removing one column
X = X[:,1:]

#Spilitting dataset into training set and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)

#feature scaling
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

#Fitting MultiLinear Regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#Preciting the Test Set
#y is dependent on x
Y_pred = regressor.predict(X_test)
 
#Building Optimal model using Backward Elimination
#import statsmodels.formula.api as sm
import statsmodels.regression.linear_model as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
X_optimal = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_optimal ).fit()
regressor_OLS.summary()

#Removing highest P value predictors
X_optimal = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_optimal ).fit()
regressor_OLS.summary()

X_optimal = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_optimal ).fit()
regressor_OLS.summary()

X_optimal = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_optimal ).fit()
regressor_OLS.summary()

X_optimal = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = Y, exog = X_optimal ).fit()
regressor_OLS.summary()

