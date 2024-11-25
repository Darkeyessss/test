import sys
import subprocess

# Install packages using subprocess, excluding tensorflow for Python 3.13 compatibility
subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "pandas", "matplotlib", "seaborn", "requests", "flask", "scikit-learn", "torch", "opencv-python-headless"])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import flask
import sklearn.datasets as datasets
from sklearn.linear_model import LinearRegression

# Sample numpy array operation
array = np.random.rand(10)
mean = np.mean(array)
print("Mean of the array:", mean)
  
# Sample pandas dataframe operation
data = {'A': np.random.rand(10), 'B': np.random.rand(10)}
df = pd.DataFrame(data)
print("DataFrame summary:")
print(df.describe())

# Sample matplotlib plot
plt.plot(array)
plt.title('Random Array Plot')
plt.xlabel('Index')
plt.ylabel('Value')
plt.show()

# Sample seaborn heatmap
sns.heatmap(df.corr(), annot=True)
plt.show()

# Sample request using requests library
response = requests.get('https://api.github.com')
print("GitHub API response status code:", response.status_code)

# Simple Flask app example
app = flask.Flask(__name__)

@app.route('/')
def home():
    return "Hello, Flask!"

# sklearn linear regression model
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=0.1)
model = LinearRegression()
model.fit(X, y)
print("Linear Regression Coefficients:", model.coef_)



# Run Flask app if needed (not running in this script)
# app.run()

print("All operations completed successfully.")