import torch.nn as nn
import torch.nn.functional as F


class CustomLoss(nn.Module):
    """
    Custom loss function wrapper.

    Args:
        config (dict): Loss function configuration with keys:
            -

    Inputs:
        pred (Tensor): Predicted values of shape (B,).
        target (Tensor): Target values of shape (B,).

    Returns:
        Tensor: Scalar loss value.
    """

    def __init__(self, config):
        super().__init__()

    def forward(self, pred, target):
        loss = F.mse_loss(pred, target)
        return loss
