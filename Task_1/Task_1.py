import matplotlib.pyplot as plt
import pandas as pd

##CALIFORNIA
dataset_California = pd.read_csv('50_Startups.csv')
z = np.arange(17)
X = z.reshape(-1, 1)
w = dataset_California.loc[dataset_California.State=='California',:]
y = w.iloc[:, -1].values

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
plt.title('California (Linear Regression)')
plt.xlabel('R&D spent')
plt.ylabel('profit')
plt.show()

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('California (Polynomial Regression)')
plt.xlabel('R&D spent')
plt.ylabel('profit')
plt.show()

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('California (Polynomial Regression)')
plt.xlabel('R&D spent')
plt.ylabel('profit')
plt.show()

lin_reg.predict([[34]])

result1 = lin_reg_2.predict(poly_reg.fit_transform([[60]]))
print ('For California')
print ( lin_reg_2.predict(poly_reg.fit_transform([[60]])))

###NEWYORK
dataset_NewYork = pd.read_csv('50_Startups.csv')
r = np.arange(17)
p = r.reshape(-1, 1)
s = dataset_NewYork.loc[dataset_NewYork.State=='New York', :]
q = s.iloc[:, -1].values

"""X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

lin_reg3 = LinearRegression()
lin_reg3.fit(p, q)

poly_reg1 = PolynomialFeatures(degree = 4)
p_poly = poly_reg1.fit_transform(p)
poly_reg1.fit(p_poly, q)
lin_reg_4 = LinearRegression()
lin_reg_4.fit(p_poly, q)

plt.scatter(p, q, color = 'red')
plt.plot(p, lin_reg3.predict(p), color = 'blue')
plt.title('Newyork(Linear Regrression)')
plt.xlabel('R&D spent')
plt.ylabel('profit')
plt.show()

plt.scatter(p, q, color = 'red')
plt.plot(p, lin_reg_2.predict(poly_reg1.fit_transform(p)), color = 'blue')
plt.title('Newyork (Polynomial Regression)')
plt.xlabel('R&D spent')
plt.ylabel('profit')
plt.show()

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_4.predict(poly_reg1.fit_transform(X_grid)), color = 'blue')
plt.title('Newyork (Polynomial Regression)')
plt.xlabel('R&D spent')
plt.ylabel('profit')
plt.show()

lin_reg3.predict([[34]])

result2 = lin_reg_4.predict(poly_reg.fit_transform([[60]]))
print ('for newyork')
print ( lin_reg_4.predict(poly_reg.fit_transform([[60]])))
print ('for california')
print ( lin_reg_2.predict(poly_reg.fit_transform([[60]])))

if result1>result2 :
    print ('California will make more profit')
else:
    print ('New York will make more profit')







