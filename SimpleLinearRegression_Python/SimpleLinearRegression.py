#Simple Linear regression

#Importing libraries
import numpy as np #mathematical tools
import matplotlib.pyplot as plt
import pandas as pd #import dataset lib

#importing the datasets
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

#Spilitting dataset into training set and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 1/3, random_state = 0)

#feature scaling
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

#Fitting Simple Linear Regression Model to the training test
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#Predicting the Test set results
y_pred = regressor.predict(X_test)

#Visualising the training set
plt.scatter(X_train,Y_train,color = 'red') #real in red
plt.plot(X_train, regressor.predict(X_train), color = 'blue') #prediction in blue
plt.title('Salary vs Expected Salary')
plt.xlabel('Years of Exp')
plt.ylabel('Salary')
plt.show()


