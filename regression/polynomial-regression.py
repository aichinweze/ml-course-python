# http://polynomialregression.drque.net/math.html - a great explanation of polynomial regression
# https://machinelearningmastery.com/gradient-descent-for-machine-learning/ - great explanation of gradient descent

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(42)
n_samples = 100

# returns 100 evenly spaced numbers between 0 and 10
X = np.linspace(0, 10, 100)

# model for noise
rng = np.random.randn(n_samples) * 100

y = X ** 3 + rng + 100

plt.figure(figsize=(12, 8))
plt.scatter(X, y)
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

lr = LinearRegression()
lr.fit(X.reshape(-1, 1), y)
model_pred = lr.predict(X.reshape(-1, 1))

plt.figure(figsize=(10, 8))
plt.scatter(X, y)
plt.plot(X, model_pred)
plt.show()

print(r2_score(y, model_pred))


from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=2)

# produces a matrix where first column is all ones and all columns
# in the top row (except first column) is zero. The second column contains
# values of X^1 in the rows. The third column will have the values of
# X^2 in the rows. The fourth column will have the values of X^3 in the
# rows etc.
X_poly = poly_reg.fit_transform(X.reshape(-1, 1))

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y.reshape(-1, 1))
y_pred = lin_reg_2.predict(X_poly)

plt.figure(figsize=(12, 8))
plt.scatter(X, y)
plt.plot(X, y_pred)
plt.show()

# Decent r2 score for this model but not the correct degree - lucky
# it works well in this locality
print(r2_score(y, y_pred))

# Attempting with real date
from sklearn.datasets import load_boston

boston_data = load_boston()
df = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)

pd.options.display.float_format = "{:,.2f}".format

X_boston = df['DIS'].values
y_boston = df['NOX'].values

plt.figure(figsize=(12, 8))
plt.scatter(X_boston, y_boston)
plt.show()

# Real data - linear regression
lr = LinearRegression()
lr.fit(X_boston.reshape(-1, 1), y_boston)
model_pred_boston = lr.predict(X_boston.reshape(-1, 1))

plt.figure(figsize=(10, 8))
plt.scatter(X_boston, y_boston)
plt.plot(X_boston, model_pred_boston)
plt.show()

print(r2_score(y_boston, model_pred_boston))

# Real data - quadratic regression
poly_reg_boston = PolynomialFeatures(degree=2)
X_poly_boston = poly_reg_boston.fit_transform(X_boston.reshape(-1, 1))
lin_reg_2_boston = LinearRegression()

lin_reg_2_boston.fit(X_poly_boston, y_boston.reshape(-1, 1))

# Gives us a linearly spaced set of values for X. Concern in the model here
# seems to be more about getting right shape of our model curve
X_fit = np.arange(X_boston.min(), X_boston.max(), 1)[:, np.newaxis]
y_pred = lin_reg_2_boston.predict(poly_reg_boston.fit_transform(X_fit.reshape(-1, 1)))

plt.figure(figsize=(12, 8))
plt.scatter(X_boston, y_boston)
plt.plot(X_fit, y_pred)
plt.show()

print(r2_score(y_boston, lin_reg_2_boston.predict(X_poly_boston)))

# Plots show that the model is good for certain ranges of values in our predictors.
# Outside of that, we should consider either using another model or not using ML.

# Real data - cubic regression
poly_reg_b3 = PolynomialFeatures(degree=3)
X_poly_b3 = poly_reg_b3.fit_transform(X_boston.reshape(-1, 1))
lin_reg_3_boston = LinearRegression()

lin_reg_3_boston.fit(X_poly_b3, y_boston.reshape(-1, 1))
y_pred = lin_reg_3_boston.predict(poly_reg_b3.fit_transform(X_fit.reshape(-1, 1)))

plt.figure(figsize=(12, 8))
plt.scatter(X_boston, y_boston)
plt.plot(X_fit, y_pred)
plt.show()

print(r2_score(y_boston, lin_reg_3_boston.predict(X_poly_b3)))

