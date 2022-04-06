import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn

import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.datasets import load_boston
from sklearn.metrics import r2_score

boston_data = load_boston()
df = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)

print(df.head())
print(df.shape)

X = df
# Target populates our dependent variable y. In this case it should take the MEDV column from the data set
y = boston_data.target

# Add a separate constant term so it is not incorporated in the model ie y = mX rather than y = mX + c
# Should just be a column of ones (and then these are scaled with a coefficient I guess?)
X_with_constant = sm.add_constant(X)
print(X_with_constant)

# Endogenous = Dependent variable
# Exogenous = Independent variable
model = sm.OLS(y, X_with_constant)
lr = model.fit()

print(lr.summary())

# smf does the same thing as the sm library but makes it MUCH easier to select the columns you use to create your model
form_lr = smf.ols(formula='y ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT',
                  data=df)
mlr = form_lr.fit()

print(mlr.summary())

# I want to remove INDUS and AGE from the model as their significance level is above the threshold (0.005)
adjusted_form_lr = smf.ols(formula='y ~ CRIM + ZN + CHAS + NOX + RM + DIS + RAD + TAX + PTRATIO + B + LSTAT',
                           data=df)
adjusted_mlr = adjusted_form_lr.fit()

print(adjusted_mlr.summary())

# ----------------------------------------------------------------------------------
# EXERCISE: Create a model with the following features: CRIM, ZN, CHAS, NOX
exercise_lr = smf.ols(formula='y ~ CRIM + ZN + CHAS + NOX',
                      data=df)
exercise_mlr = exercise_lr.fit()
print(exercise_mlr.summary())

# Forces pandas to autodetect size of terminal window so you can see all columns
pd.options.display.width = 0
# Sets the float format of numbers displayed through pandas
pd.options.display.float_format = '{:,.2f}'.format
# ----------------------------------------------------------------------------------

# Investigating collinearity
# METHOD 1: CORRELATION MATRIX
corr_matrix = df.corr()
print(corr_matrix)

# Applying a mask so we can focus on important bits
corr_matrix[np.abs(corr_matrix) < 0.6] = 0
print(corr_matrix)

plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu')
plt.show()

# METHOD 2: EIGENVALUES AND EIGENVECTORS
pd.options.display.float_format = '{:,.4f}'.format

eigenvalues, eigenvectors = np.linalg.eig(df.corr())

print(pd.Series(eigenvalues).sort_values())
# NB in output from above, column 8 has an eigenvalue of 0.0635. This is near to zero/v small compared to others. The
# small value indicates the presence of collinearity

print(np.abs(pd.Series(eigenvectors[:, 8])).sort_values(ascending=False))
# The above shows that columns 9, 8 and 2 have very high loading when compared against the rest - these are the factors
# causing the multi-collinearity problem
print(df.columns[2], df.columns[8], df.columns[9])

# The scales of values in the histograms are on different orders of magnitude => difficult to compare
plt.hist(df['TAX'])
plt.show()
plt.hist(df['NOX'])
plt.show()

# Complete version of the lr model
lr = smf.ols(formula='y ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT',
             data=df)
benchmark = lr.fit()
print(r2_score(y, benchmark.predict(df)))

# Version of lr model without LSTAT
lr = smf.ols(formula='y ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B',
             data=df)
lr_without_LSTAT = lr.fit()
print(r2_score(y, lr_without_LSTAT.predict(df)))

# Version of lr model without AGE
lr = smf.ols(formula='y ~ CRIM + ZN + INDUS + CHAS + NOX + RM + DIS + RAD + TAX + PTRATIO + B + LSTAT',
             data=df)
lr_without_AGE = lr.fit()
print(r2_score(y, lr_without_AGE.predict(df)))

# Looking at the output - Without LSTAT, there is a significant change in the R^2 score, whereas without AGE the change
# is negligible. This indicates that LSTAT is a significant variable in this model and AGE is not

