import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

# Read the data
data = pd.read_csv("data_sets\cost_revenue_dirty.csv")

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
plt.show()