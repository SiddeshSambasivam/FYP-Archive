import json
from typing import Dict, Any
from timeit import default_timer as timer


import click
from dso import DeepSymbolicRegressor

from .data import load_data, load_equation, get_data
from .metrics import get_pointwise_acc, Metrics


# get_data(eq, num_of_points, mode) -> x,y 
# load_equation(path, idx) -> eq

# df = load_data()
# loop each equation in df
#   eq = load_equation()
#   x, y =  get_data(eq, num_of_points, mode)
#   model = Model()
#   model.fit(x, y)

#   NOTE: During test time - 125 points are sampled
#   y_hat = model.predict(x)

#   a1_{mode} = get_pointwise_acc(y, y_hat)



# def get_pointwise_acc(y_true, y_pred, rtol, atol):
#     # r = roughly_equal(y_true, y_pred, rel_threshold)
#     r = np.isclose(y_true, y_pred, rtol=rtol, atol=atol, equal_nan=True)
#     return r.mean()

def load_config(path: str="configs/dso_config.json") -> Dict[str, Any]:
    with open('configs/dso_config.json') as file:
        config = json.load(file)
    
    return config

def run_benchmark(data_path: str):
    
    config = load_config()

    df = load_data(data_path)
    m = Metrics()
    n_equations = 0
    n_correct_pred = 0
    for idx, row in df.iterrows():
        # if idx == 1: break
        eq = load_equation(df, idx)
        x, y = get_data(eq, row['num_points'], 'ood')
        x_test, y_test = get_data(eq, 125, 'ood')

        print(x.shape, y.shape)

        model = DeepSymbolicRegressor(config)
        model.fit(x, y)  

        start = timer()
        y_pred = model.predict(x_test)
        end = timer()
        duration = end - start

        acc = get_pointwise_acc(y_test, y_pred, atol=1e-3, rtol=0.05)
        if acc > 0.95:
            n_correct_pred += 1
        
        n_equations += 1
        print('Duration: ',)
        m.log_metric(row['eq'], acc, ('%.10f' % duration))
    
    print(f"Accuracy: {n_correct_pred}")
    print(f'Total equations: {n_equations}')
    m.to_csv("results.csv")


@click.command()
@click.option(
    "--data-path",
    "-d",
    type=str,
    help="Path to the csv containing the equations",
    required=True 
)
def main(data_path: str) -> None:
    run_benchmark(data_path)

if __name__ == "__main__":
    main()
