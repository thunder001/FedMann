import torch
from dataset.get_dataset import get_dataloader
from model.model_evaluate import model_evaluate

import torch.nn as nn
import torch.optim as optim
import pdb


def get_loss(logits, y, beta_local, local_reg, global_reg):
    """
    Calculate the local loss for the model.
    Args:
        beta_shared: Shared model parameters.
        beta_local: Local model parameters.
        lambda_: Regularization parameter.
    Returns:
        Local loss value.
    """
    # breakpoint()
    loss = nn.BCEWithLogitsLoss()(logits, y)
    if global_reg is not None:
        local_loss = loss + global_reg * torch.sum((beta_local) ** 2) / len(y)
    else:
        local_loss = loss + local_reg * torch.sum((beta_local) ** 2) / len(y)
    return local_loss


def train(trainset, model, strategy='shared_only', local_reg=0.01, global_reg=1, epochs=10):
    """
    Train the logistic regression model on the provided training data, logging
    training loss per epoch.
    Args:
        trainset: Training dataset (DataFrameDataset).
        model: The model to train.
        strategy (str): 'shared_only' or 'full'.
        local_reg (float): L2 penalty on local-only variables (used when strategy='full'
            and global_reg is None).
        global_reg (float): L2 penalty on all variables.
        epochs (int): Number of training epochs.
    Returns:
        tuple: (model.state_dict(), trainset sample size)
    """
    # breakpoint()

    # Identify local variable indices based on the data source, assuming variables are in a fixed order
    # and align with the model's input features and linear layer weights.

    feature_cols = trainset.get_features()
    if strategy == 'full':
        local_cols = trainset.get_local_cols()
        local_cols_idx = [feature_cols.index(col) for col in local_cols]

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    train_dataloader = get_dataloader(trainset, batch_size=64, shuffle=True, one_batch=False)

    for epoch in range(epochs):
        model.train()
        losses = []
        train_data_iter = iter(train_dataloader)
        for batch in train_data_iter:
            optimizer.zero_grad()
            batch_X = batch['features']
            batch_y = batch['target'].view(-1, 1)
            logits = model(batch_X)
            if strategy == 'shared_only':
                if global_reg is not None:
                    beta_local = model.state_dict()['linear.weight'][0]
                else:
                    beta_local = torch.tensor([0.0], requires_grad=False)
            elif strategy == 'full':
                if global_reg is not None:
                    beta_local = model.state_dict()['linear.weight'][0]
                else:
                    beta_local = model.state_dict()['linear.weight'][0][local_cols_idx]
            loss = get_loss(logits, batch_y, beta_local, local_reg, global_reg)
            losses.append(loss)
            loss.backward()
            optimizer.step()
        epoch_loss = torch.mean(torch.tensor(losses))
        print(f"Epoch {epoch+1}/{epochs}, Training loss: {epoch_loss:.4f}")

    return model.state_dict(), trainset.__len__()
