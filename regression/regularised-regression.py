import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

import pandas as pd

# LINEAR REGRESSION handling outliers
from sklearn.linear_model import LinearRegression

np.random.seed(42)
n_samples = 100
rng = np.random.randn(n_samples) * 10
y_gen = 0.5 * rng + 2 * np.random.randn(n_samples)

lr = LinearRegression()
lr.fit(rng.reshape(-1, 1), y_gen)
model_pred = lr.predict(rng.reshape(-1, 1))

plt.figure(figsize=(10, 8))
plt.scatter(rng, y_gen)
plt.plot(rng, model_pred)
plt.show()

# Coefficient currently at ~ 0.471
print("Coefficient Estimate: ", lr.coef_)

# Adding random outliers
idx = rng.argmax()
y_gen[idx] = 200
idx = rng.argmin()
y_gen[idx] = -200

o_lr = LinearRegression(normalize=True)
o_lr.fit(rng.reshape(-1, 1), y_gen)
o_model_pred = o_lr.predict(rng.reshape(-1, 1))

plt.scatter(rng, y_gen)
plt.plot(rng, o_model_pred)
plt.show()

# Coefficient currently at ~ 1.506 => Outlier has caused the model to shift quite drastically
print("Coefficient Estimate: ", o_lr.coef_)

# RIDGE REGRESSION handling outliers
from sklearn.linear_model import Ridge

ridge_mod = Ridge(alpha=0.5, normalize=True)
ridge_mod.fit(rng.reshape(-1, 1), y_gen)
ridge_model_pred = ridge_mod.predict(rng.reshape(-1, 1))

plt.figure(figsize=(10, 8))
plt.scatter(rng, y_gen)
plt.plot(rng, ridge_model_pred)
plt.show()

# Coefficient currently at ~ 1.004 => Coefficient affected by outliers less than in linear regression model
print("Coefficient Estimate: ", ridge_mod.coef_)


# LASSO REGRESSION handling outliers
from sklearn.linear_model import Lasso

lasso_mod = Lasso(alpha=0.4, normalize=True)
lasso_mod.fit(rng.reshape(-1, 1), y_gen)
lasso_model_pred = lasso_mod.predict(rng.reshape(-1, 1))

plt.figure(figsize=(10, 8))
plt.scatter(rng, y_gen)
plt.plot(rng, lasso_model_pred)
plt.show()

# Coefficient currently at ~ 1.063 => Coefficient affected by outliers less than in linear regression model
print("Coefficient Estimate: ", lasso_mod.coef_)


# ELASTIC NET handling outliers
from sklearn.linear_model import ElasticNet

en_mod = ElasticNet(alpha=0.02, normalize=True)
en_mod.fit(rng.reshape(-1, 1), y_gen)
en_model_pred = en_mod.predict(rng.reshape(-1, 1))

plt.figure(figsize=(10, 8))
plt.scatter(rng, y_gen)
plt.plot(rng, en_model_pred)
plt.show()

# Coefficient currently at ~ 0.747 => Least affected by outliers - better than Lasso and Ridge regression here
print("Coefficient Estimate: ", en_mod.coef_)


