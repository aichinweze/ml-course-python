import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import seaborn as sns

print(sklearn.__version__)

path_to_housing_data = 'C://Users//IfeanyiChinweze//Projects//training//ml-course-python//ml-course-data//data//housing.data'
df = pd.read_csv(path_to_housing_data, delim_whitespace=True, header=None)

col_name = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.columns = col_name

print(df.head())
print(df.describe())

# We want to ensure that the statistics in our feature columns are similar. This will be revisited later

col_study = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM']
sns.pairplot(df[col_study], height=2.5)
# RM against itself gives a nice, normal distribution
# ZN and CRIM against themselves give a single-tailed distribution - be wary, this behaviour is harder to model
plt.show()

# Correlation Analysis
# corr() function produces an NxN matrix with correlation between each of the feature columns
print(df.corr())

plt.figure(figsize=(16, 10))
sns.heatmap(df.corr(), annot=True)
plt.show()

