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


