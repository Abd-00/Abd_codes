import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('global_co2.csv')
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1].values

"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Global CO2')
plt.xlabel('Years')
plt.ylabel('Total CO2 Production')
plt.show()

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Global CO2')
plt.xlabel('Years')
plt.ylabel('Total CO2 Production')
plt.show()

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Global CO2')
plt.xlabel('Years')
plt.ylabel('Total CO2 Production')
plt.show()

y_pred1 = lin_reg_2.predict(poly_reg.fit_transform([[2011]]))
y_pred2 = lin_reg_2.predict(poly_reg.fit_transform([[2012]]))
y_pred3 = lin_reg_2.predict(poly_reg.fit_transform([[2013]]))
print(y_pred1)
print(y_pred2)
print(y_pred3)