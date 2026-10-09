import torch
import sys
import torch.nn as nn
import numpy as np


class LogisticRegressionModel(nn.Module):
    """
    A class to encapsulate the logistic regression model.
    """
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)
    
    def forward(self, x):
        """Forward pass for the logistic regression model.
        
        Args:
            x (torch.Tensor): Input tensor. 
        Returns:
            torch.Tensor: Output tensor after applying the linear layer.
        """
        return self.linear(x)
