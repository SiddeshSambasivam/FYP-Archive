# Completed components
# 1. Dataset class 
# 2. Nemo Model

# TODO:
# 1. Trainer loop
# 2. Train/test split - Holdout method

# Q. How does training work?
# During training, the model learns operator identification from the training data.

# Q. How does testing work?
# During testing, the model takes the test data and predicts the occurrence of each operator.

# optimizer - Adam
# loss - BCEWithLogitsLoss

import logging
import click

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import src.models.Nemo.model as M
from src.dataset import SymbolicOperatorDataset

logger = logging.getLogger(__name__)

def test(model, test_loader, device):
    """
    Test the model on the test data.
    """

    with torch.no_grad():

        tp, fp, tn, fn = 0, 0, 0, 0

        for batch in test_loader:
            data, target = batch["inputs"].to(device), batch["labels"].to(device)
            output = model(data)

            pred = output.round()

            confusion_vector = pred / target
            tp += torch.sum(confusion_vector == 1).item()
            fp += torch.sum(confusion_vector == float('inf')).item()
            tn += torch.sum(torch.isnan(confusion_vector)).item()
            fn += torch.sum(confusion_vector == 0).item()

        acc = (tp + tn) / (tp + fp + tn + fn)

    return acc, tp, fp, tn, fn

def train(model, train_loader, test_loader, epochs, lr, device):
    """
    Train the model for a number of epochs.
    """

    model.to(device)
    model.train()

    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        print("\n===== Epoch {} =====".format(epoch+1))
        for batch in tqdm(train_loader):

            data, target = batch["inputs"].to(device), batch["labels"].to(device)

            optimizer.zero_grad()
            output = model(data)

            loss = loss_func(output, target)
            loss.backward()

            optimizer.step()

        acc, tp, fp, tn, fn = test(model, test_loader, device)

        print("Evaluation metrics")
        print("-------------------")
        print(f"Loss: {loss.item()} Accuracy: {acc:.2f}")
        print(f"TP: {tp} FP: {fp} TN: {tn} FN: {fn}\n")
        print(f"Precision: {tp/(tp+fp):.2f}")
        print(f"Recall: {tp/(tp+fn):.2f}")
        print(f"F1: {2*tp/(2*tp+fp+fn):.2f}")

@click.command()
@click.option(
    "--data-path",
    "-d",
    type=str,
    help="Path to the csv containing the equations",
    required=True,
)
@click.option(
    "--noise",
    "-n",
    type=float,
    help="Gaussian noise to add to the data",
    default=0.0,
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    help="Batch size for training",
    default=1,
)
@click.option(
    "--epochs",
    "-e",
    type=int,
    help="Number of epochs to train the model",
    default=10,
)
@click.option(
    "--lr",
    "-l",
    type=float,
    help="Learning rate",
    default=0.001,
)
@click.option(
    "--device",
    "-d",
    type=str,
    help="Device to use for training",
    default="cpu",
)
@click.option(
    "--output-path",
    "-o",
    type=str,
    help="Path to save the model",
    default="model.pt",
)
def main(data_path: str, noise: float, batch_size:int, epochs: int, lr: float, device: str, output_path:str) -> None:
    """
    Train the model.
    """
    
    seed = 1
    torch.manual_seed(seed)    
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True    

    dataset = SymbolicOperatorDataset(data_path, noise=noise, max_variables=5)
    
    train_size = int(0.80 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = M.NemoModel(
        dim_in=5,
        num_classes=10,
        num_heads=2,
        num_seed_features=50,
        num_inds=40,
        ln=True,
        dim_hidden=256
        )    

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    logger.log(logging.INFO, "Training model")
    train(model, train_loader, test_loader, epochs, lr, device)
    
    logger.log(logging.INFO, "Training complete")
    
    # logger.log(logging.INFO, "Saving model")
    # torch.save(model.state_dict(), output_path)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
    






