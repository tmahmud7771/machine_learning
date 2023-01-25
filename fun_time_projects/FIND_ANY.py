import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

# Read the data
print("Enter the path of the data set")
path = f"..\data_sets\{input()}"
print(path)
data = pd.read_csv(path)

target =  input("Enter the target variable name: ")
feature = input("Enter the feature variable name: ")

title = input("Enter the title of the graph: hint (Film Cost vs Global Revenue)")


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



X = DataFrame(data, columns=[target]) # target variable or dependent variable
y = DataFrame(data, columns=[feature]) # feature variable or independent variable

plt.figure(figsize=(10,6))
plt.scatter(X, y)
plt.title('Film Cost vs Global Revenue')
plt.xlabel('Production Budget')
plt.ylabel('Worldwide Gross')
plt.ylim(0, 3000000000)
plt.xlim(0, 450000000)
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
plt.title(title)
plt.xlabel(target)
plt.ylabel(feature)
plt.ylim(0, 3000000000)
plt.xlim(0, 450000000)
plt.show()

m = regression.coef_[0][0]
c = regression.intercept_[0]


print("The equation of the regression line is y = {m}x + {c}".format(m=m, c=c)) 

print("Enter the value for your prediction")

budget = float(input())

y_val = m * budget + c

print("The predicted value is  {y_val}".format(y_val=y_val))

# regression score

print("Regression Score=",regression.score(X, y))


