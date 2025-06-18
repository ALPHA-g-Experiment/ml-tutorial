import numpy as np
import torch


def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
    """
    Run one training epoch.

    Args:
        dataloader (DataLoader): PyTorch DataLoader yielding (input, target) batches.
        model (nn.Module): PyTorch model to train.
        loss_fn (callable): Loss function taking (pred, target) and returning a scalar loss Tensor.
        optimizer (Optimizer): PyTorch optimizer.
        device (torch.device): Device to perform computations on.

    Returns:
        float: Average loss over the epoch.
    """
    model.train()

    total_loss = 0.0
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.detach().item()

    return total_loss / len(dataloader)


def test_one_epoch(dataloader, model, loss_fn, device):
    """
    Evaluate the model on a validation set.

    Args:
        dataloader (DataLoader): PyTorch DataLoader yielding (input, target) batches.
        model (nn.Module): PyTorch model to evaluate.
        loss_fn (callable): Loss function taking (pred, target) and returning a scalar loss Tensor.
        device (torch.device): Device to perform computations on.

    Returns:
        tuple: A tuple containing:
            - float: Average loss over the validation set.
            - float: Mean of (5% trimmed) residuals.
            - float: Standard deviation of (5% trimmed) residuals.
    """
    model.eval()

    total_loss = 0.0
    residuals = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            total_loss += loss_fn(pred, y).item()
            residuals = np.append(residuals, (pred - y).cpu().numpy())
    total_loss /= len(dataloader)

    residuals = np.sort(residuals)
    n = len(residuals)
    trimmed_residuals = residuals[int(0.05 * n) : int(0.95 * n)]

    return total_loss, trimmed_residuals.mean(), trimmed_residuals.std(ddof=1)
