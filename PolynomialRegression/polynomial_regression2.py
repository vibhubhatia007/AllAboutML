#importing the library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#fitting a linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#fitting the Polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)

#visualling Linear regression results
plt.scatter(X,y, color = 'black')
plt.plot(X, lin_reg.predict(X), color = 'black')
plt.title('Truth or Bluf(Linear Regression)')
plt.xlabel('Position')
plt.ylabel('Salaries')
plt.show()

#visuallising linear regression results
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid),1))

plt.scatter(X,y, color = 'green')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff(Polynomial Regression)')
plt.xlabel('Position')
plt.ylabel('Salaries')
plt.show()
#also improve the code for resolution issues
''' put in the following codes in line number 32 onwards
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = np.reshape((len(X_grid),1))
also change the parameter X by X_grid 
next use the predict method to predict the salary of any employee
just change the parameter from X to any number of level
lin_reg.predict(5)
lin_reg2.predict(poly_reg.fit_transform(5))'''

