import numpy as np

# Larger synthetic dataset
np.random.seed(0)
n = 100  # Number of data points
X1 = np.random.uniform(0, 10, n)
X2 = np.random.uniform(0, 10, n)
beta_0 = 2
beta_1 = 3
beta_2 = 4
sigma = 2
Y = beta_0 + beta_1 * X1 + beta_2 * X2 + np.random.normal(0, sigma, n)

# Save the dataset to a CSV file
import pandas as pd
data = pd.DataFrame({'Y': Y, 'X1': X1, 'X2': X2})
data.to_csv('bayesian_regression_data.csv', index=False)
