import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('monthlyexp vs incom.csv')
x = dataset.iloc[:,0:1].values
y = dataset.iloc[:,1:2].values
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)


x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color = 'blue')
plt.title('Experience vs Income')
plt.xlabel('Monthly Experience')
plt.ylabel('Income')
plt.show()






