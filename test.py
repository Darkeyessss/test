import numpy as np
import pandas as pd

# Sample numpy array operation
array = np.random.rand(10)
mean = np.mean(array)
print("Mean of the array:", mean)

# Sample pandas dataframe operation
data = {'A': np.random.rand(10), 'B': np.random.rand(10)}
df = pd.DataFrame(data)
print("DataFrame summary:")
print(df.describe())