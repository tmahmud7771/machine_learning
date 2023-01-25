import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

# Read the data
data = pd.read_csv("..\data_sets\cost_revenue_dirty.csv")

# Print out the first rows of data
print(data.head())

# Print out the last rows of data
print(data.tail())

# Print out the shape of data
print(data.shape)

# Print out the column names of data

print(data.columns)

# Print out the data types of data
print(data.dtypes)

# Print out the summary statistics of data
print(data.describe())

X = DataFrame(data, columns=['production_budget']) # target variable or dependent variable
y = DataFrame(data, columns=['worldwide_gross']) # feature variable or independent variable

plt.figure(figsize=(10,6))
plt.scatter(X, y)
plt.title('Film Cost vs Global Revenue')
plt.xlabel('Production Budget')
plt.ylabel('Worldwide Gross')
plt.ylim(0, 3000000000)
plt.xlim(0, 450000000)
plt.style.use('fivethirtyeight')
plt.show()

# Create linear regression object
regression = LinearRegression()

# Train the model using the training sets
regression.fit(X, y)

# The coefficients
print('Coefficients: \n', regression.coef_)

# the intercept
print('Intercept: \n', regression.intercept_)


# Plot outputs


plt.figure(figsize=(10,6))
plt.scatter(X, y)
plt.plot(X, regression.predict(X), color='red', linewidth=4)
plt.title('Film Cost vs Global Revenue')
plt.xlabel('Production Budget')
plt.ylabel('Worldwide Gross')
plt.ylim(0, 3000000000)
plt.xlim(0, 450000000)
plt.style.use('fivethirtyeight')
plt.show()

m = regression.coef_[0][0]
c = regression.intercept_[0]


print("The equation of the regression line is y = {m}x + {c}".format(m=m, c=c)) 

print("Enter the production budget of the film to predict the revenue")

budget = float(input())

y_val = m * budget + c
print("---------------------------------------------------------------")
print("The predicted revenue of the film is {y_val}".format(y_val=y_val))
print("---------------------------------------------------------------")

# regression score
print("---------------------------------------------------------------")
print("Regression Score:",regression.score(X, y))
print("---------------------------------------------------------------")


