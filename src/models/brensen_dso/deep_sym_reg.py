import pdb
from dso import DeepSymbolicRegressor
import numpy as np

# Generate some data
np.random.seed(0)
X = np.random.random((10, 2))
y = np.sin(X[:, 0]) + np.cos(X[:, 1])

# Create the model
model = (
    DeepSymbolicRegressor()
)  # Alternatively, you can pass in your own config JSON path

# Fit the model
model.fit(X, y)  # Should solve in ~10 seconds

# View the best expression
print(model.program_.pretty())

# Make predictions
out = model.predict(2 * X)

import pdb

pdb.set_trace()
