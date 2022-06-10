import pdb
import json
from typing import Any, Dict
from dso import DeepSymbolicRegressor
import numpy as np

def load_config(path: str="configs/dso_config.json") -> Dict[str, Any]:
    with open('configs/dso_config.json') as file:
        config = json.load(file)
    
    return config

# def init_model():
# Generate some data
np.random.seed(0)

def f(x):
    return np.sin(x[:, 0]) + np.cos(x[:, 1]) + 1 

X = np.random.random((200, 2))
y = f(X)
# + np.exp(X[:,0]*X[:,1])

config = load_config()

# Create the model
model = (
    DeepSymbolicRegressor(config)
)  # Alternatively, you can pass in your own config JSON path

# Fit the model
model.fit(X, y)  # Should solve in ~10 seconds

# View the best expression
print(model.program_.pretty())

# Make predictions

x_test = np.random.random((150, 2))
y_test = f(x_test)

y_pred = model.predict(x_test)

r = np.isclose(y_test, y_pred,  atol=1e-3, rtol=0.05, equal_nan=True)
# print(r)
mean = r.mean()

print('Mean: ', mean)

import pdb

pdb.set_trace()
