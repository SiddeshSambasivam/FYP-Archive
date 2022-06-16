from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from sklearn.utils.random import check_random_state

import numpy as np

from time import process_time

from tqdm import tqdm

# Ground truth
x0 = np.arange(-1, 1, .1)
x1 = np.arange(-1, 1, .1)
x0, x1 = np.meshgrid(x0, x1)
y_truth = x0**2 - x1**2 + x1 - 1

rng = check_random_state(0)
# duration = 0

# Training samples
def exp(x1):
    with np.errstate(over='ignore'):
        return np.where(np.abs(x1) < 100, np.exp(x1), 0.)    

exp = make_function(function=exp, name='exp',arity=1)

acc = 0
duration = 0

for i in tqdm(range(10)):

    X_train = rng.uniform(-1, 1, 20000).reshape(10000, 2)
    y_train = X_train[:, 0]**2 - X_train[:, 1]**2 + X_train[:, 1] - 1

    # Testing samples
    X_test = rng.uniform(-1, 1, 256).reshape(128, 2)
    y_test = X_test[:, 0]**2 - X_test[:, 1]**2 + X_test[:, 1] - 1

# {+, −, ×, ÷,
    # √, ln, exp, neg, inv,sin, cos}g
    est_gp = SymbolicRegressor(population_size=2**10,
                            const_range=(-4*np.pi, 4*np.pi),
                            tournament_size=20,
                            function_set=(
                                'add', 'sub', 'mul', 'div', 'sqrt', 'log', exp, 'inv', 'sin', 'cos'),
                            generations=20, verbose=1)
    est_gp.fit(X_train, y_train)

    # y_gp = est_gp.predict(np.c_[x0.ravel(), x1.ravel()]).reshape(x0.shape)

    start = process_time()
    y_pred = est_gp.predict(X_test)
    end = process_time()
    duration += (end-start)

    r = np.isclose(y_test, y_pred, atol=1e-3, rtol=0.05, equal_nan=True)
    r = r.mean()

    if r > 0.95:
        acc += 1

    print(est_gp._program, r)

# score_gp = est_gp.score(X_test, y_test)
print(f'Accuracy: {acc}\t Duration: {duration}')
# print(y_pred, y_test)