import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

path_to_housing_data = 'C://Users//IfeanyiChinweze//Projects//training//ml-course-python//ml-course-data//data//housing.data'
df = pd.read_csv(path_to_housing_data, delim_whitespace=True, header=None)

col_name = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.columns = col_name

print(df.head())

# RM (Average number of rooms per dwelling)
X = df['RM'].values.reshape(-1, 1)

# MEDV (Median value of owner-occupied homes)
y = df['MEDV'].values

from sklearn.linear_model import RANSACRegressor

ransac = RANSACRegressor()

ransac.fit(X, y)

# The masks are made when we fit the model to the data
# Inlier mask appears to be the array of values that have been determined not to be outliers
inlier_mask = ransac.inlier_mask_
# Outlier mask is what is not contained within the inlier mask ie the outliers
outlier_mask = np.logical_not(inlier_mask)

# Create an array that goes from 3 up to 10 in increments of 1
incs = np.arange(3, 10, 1)
print(incs)

line_X = incs
line_Y_ransac = ransac.predict(line_X.reshape(-1, 1))

# Beta coefficient and y-intercept for model
print("Ransac coefficients:")
print(ransac.estimator_.coef_)
print(ransac.estimator_.intercept_)

# Linear Regression model for comparison
from sklearn.linear_model import LinearRegression
linear = LinearRegression()
linear.fit(X, y)

line_Y_linear = linear.predict(line_X.reshape(-1, 1))

sns.set(style="darkgrid", context="notebook")
plt.figure(figsize=(12, 8))
plt.scatter(X[inlier_mask], y[inlier_mask], c="blue", marker="o", label="Inliers")
plt.scatter(X[outlier_mask], y[outlier_mask], c="brown", marker="s", label="Outliers")
plt.plot(line_X, line_Y_ransac, color="red")
plt.plot(line_X, line_Y_linear, color="green")
plt.xlabel("Average number of rooms per dwelling")
plt.ylabel("Median value of owner occupied homes ($1000s)")
plt.legend(loc="upper left")
plt.show()


# EVALUATING THE MODEL
from sklearn.model_selection import train_test_split

# SLICE NOTATION
# This use of colons in this array is referred to as slice notation in python
# ":" is in the row position and indicates that we should take every row of the array
# ":-1" means take everything except for the last item (ie do not include the last column). We do this because that
# column becomes our y vector
# Slice operator works like this array[start:stop:step] or where parts are omitted:
# a[start:stop]  = items start through stop-1
# a[start:]      = items start through the rest of the array
# a[:stop]       = items from the beginning through stop-1
# a[:]           = a copy of the whole array

# For example, a[1::-1] = first 2 items but order reversed; a[-3::-1] = everything but last 2 items but order reversed

X = df.iloc[:, :-1].values
y = df['MEDV'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

robust = RANSACRegressor()

robust.fit(X_train, y_train)

y_train_pred = robust.predict(X_train)
y_test_pred = robust.predict(X_test)


# Method 1: Residual Analysis
def residual_analysis(observed_training_val, pred_train_val, observed_test_val, pred_test_val):
    plt.figure(figsize=(12, 8))
    plt.scatter(pred_train_val, pred_train_val - observed_training_val, c='blue', marker='o', label='Training data')
    plt.scatter(pred_test_val, pred_test_val - observed_test_val, c='orange', marker='*', label='Test data')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='k')
    plt.xlim([-10, 50])
    plt.show()


residual_analysis(y_train, y_train_pred, y_test, y_test_pred)

# Method 2: Mean Squared Error (MSE)
from sklearn.metrics import mean_squared_error

mse_y_train = mean_squared_error(y_train, y_train_pred)
mse_y_test = mean_squared_error(y_test, y_test_pred)

print("Mean Squared Errors: ")
print(mse_y_train)
print(mse_y_test)

# Method 3: Coefficient of Determination, R^2
from sklearn.metrics import r2_score

r2_y_train = r2_score(y_train, y_train_pred)
r2_y_test = r2_score(y_test, y_test_pred)

print("R2 score:")
print(r2_y_train)
print(r2_y_test)

# A near perfect model
generate_random = np.random.RandomState(0)
x = 10 * generate_random.rand(1000)
y = 3 * x + np.random.rand(1000)

plt.figure(figsize=(10, 8))
plt.scatter(x, y)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# fit_intercept=False sets the y-intercept to 0; fit_intercept=True fits the y-intercept along the line of best fit
model = LinearRegression(fit_intercept=True)
model.fit(X_train.reshape(-1, 1), y_train)

y_train_pred = model.predict(X_train.reshape(-1, 1))
y_test_pred = model.predict(X_test.reshape(-1, 1))

residual_analysis(y_train, y_train_pred, y_test, y_test_pred)

mse_y_train = mean_squared_error(y_train, y_train_pred)
mse_y_test = mean_squared_error(y_test, y_test_pred)

print("Mean Squared Errors: ")
print(mse_y_train)
print(mse_y_test)

r2_y_train = r2_score(y_train, y_train_pred)
r2_y_test = r2_score(y_test, y_test_pred)

print("R2 score:")
print(r2_y_train)
print(r2_y_test)

