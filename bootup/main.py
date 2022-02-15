# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
# Statistical data visualisation library
import seaborn as sns
import matplotlib.pyplot as plt


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
print_hi("World")
print(1+1)

var = 5
print(var)

# ** is a power operator
print(2**4)

print(np.__version__)
print(pd.__version__)
print(sns.__version__)

import sys
print(sys.version)

# There are no official headers so it assumes the first row is a header - untrue. Hence, why we add the header=None line
df = pd.read_csv('data//iris.data')
# df.head gives the first 5 rows of the dataframe
print(df.head())

df = pd.read_csv('data//iris.data', header=None)
print(df.head())

# Give the dataframe columns
col_name = ["sepal length", "sepal width", "petal length", "petal width", "class"]
df.columns = col_name
print(df.head())

# Seaborn has some data sets by default such as the iris one loaded above
iris = sns.load_dataset("iris")
print("Seaborn dataset")
print(iris.head())

# Provides some basic statistics for the dataset
print("Read dataset")
print(df.describe())
print("Loaded dataset")
print(iris.describe())

print("Dataset basic info")
print(iris.info())

print(iris.groupby("species").size())
print(df.groupby("class").size())


# pairplot is colour coded
sns.pairplot(iris, hue="species", height=3, aspect=1)
# plt.show needed to see the plot created by the line below
plt.show()

iris.hist(edgecolor="black", linewidth=1.2, figsize=(12,8))
plt.show()

# White dot in violin plot is the mean
plt.figure(figsize=(12,8))
plt.subplot(2, 2, 1)
sns.violinplot(x="species", y="sepal_length", data=iris)
plt.subplot(2, 2, 2)
sns.violinplot(x="species", y="sepal_width", data=iris)
plt.subplot(2, 2, 3)
sns.violinplot(x="species", y="petal_length", data=iris)
plt.subplot(2, 2, 4)
sns.violinplot(x="species", y="petal_width", data=iris)
plt.show()

iris.boxplot(by="species", figsize=(12, 8))
plt.show()

pd.plotting.scatter_matrix(iris, figsize=(12, 10))
plt.show()

sns.pairplot(iris, hue="species", diag_kind="kde")
plt.show()
