#Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

#Data too short to have separate test sets and a training set

"""Two models created for comparision only"""

# Fitting Linear regression to dataset
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X,Y)

#Fitting Polynomial regression to dataset
from sklearn.preprocessing import PolynomialFeatures
polynomial_regressor = PolynomialFeatures(degree = 8)
X_polynomial = polynomial_regressor.fit_transform(X)
#New linear regression object fitted with polynomial X with original Y
linear_regressor2 = LinearRegression()
linear_regressor2.fit(X_polynomial,Y)

#Visualisation of linear model results
plt.scatter(X,Y, color = 'red')
plt.plot(X,linear_regressor.predict(X), color = 'blue')
plt.title('Linear Model Results') 
plt.xlabel('X Values - Designation')
plt.ylabel('Y Values - Salary')
plt.show()

#Visualisation of polynomial model results
plt.scatter(X,Y, color = 'red')
plt.plot(X, linear_regressor2.predict(polynomial_regressor.fit_transform(X)), color = 'blue')
plt.title('Polynomial Model Results')
plt.xlabel('X Values - Designation')
plt.ylabel('Y Values - Salary')
plt.show()
