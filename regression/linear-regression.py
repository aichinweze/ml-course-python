import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

path_to_housing_data = 'C://Users//IfeanyiChinweze//Projects//training//ml-course-python//ml-course-data//data//housing.data'
df = pd.read_csv(path_to_housing_data, delim_whitespace=True, header=None)

col_name = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.columns = col_name

print(df.head())

# Here we create a simple use of linear regression with only one feature, RM (Average number of rooms per dwelling)
# X is our independent variable/predictor
# reshape(-1, 1) is used to reshape to an Nx1 column vector when you don't know or want to explicitly determine the
# value of N
X = df['RM'].values.reshape(-1, 1)

# MEDV (Median value of owner-occupied homes) is the value we are trying to predict aka target variable
y = df['MEDV'].values

# ML Process Step 1: Choose a model
from sklearn.linear_model import LinearRegression

# Linear Regression Model 1: Rooms per dwelling vs Median house value
model = LinearRegression()
model.fit(X, y)

# Coefficient represents β in the regression model, which is the gradient of our regression line
print(model.coef_)
# Y-intercept of our regression line
print(model.intercept_)

plt.figure(figsize=(12,10))
sns.regplot(X, y)
plt.xlabel('average number of rooms per dwelling')
plt.ylabel('median value of owner-occupied homes in $1000s')
plt.show()

# Make predictions based on the model we have created
print(model.predict(np.array([5]).reshape(-1, 1)))
print(model.predict(np.array([7]).reshape(-1, 1)))


# Linear Regression Model 2:
# ML Process Step 2: Instantiate model
model_2 = LinearRegression()

# ML Process Step 3: Arrange data into features matrix (X) and target vector (y)
X = df['LSTAT'].values.reshape(-1, 1)
y = df['MEDV'].values

# ML Process Step 4: Fit the model to your data
model_2.fit(X, y)

plt.figure(figsize=(12, 10))
sns.regplot(X, y)
plt.xlabel('% lower status of the population')
plt.ylabel('median value of owner-occupied homes in $1000s')
plt.show()

# ML Process Step 5: Apply model to new data
print(model_2.predict(np.array([15]).reshape(-1, 1)))
