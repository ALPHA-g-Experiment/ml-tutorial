import torch


def build_optimizer(params, config):
    """
    Build a PyTorch optimizer.

    Args:
        params (iterable): Model parameters to optimize (e.g., model.parameters()).
        config (dict): Optimizer configuration with keys:
            - `lr` (float): Learning rate.
            - `betas` (tuple(float, float)): Coefficients used for computing
              running averages of gradient and its square.
            - `eps` (float): Term added to the denominator to improve numerical
              stability.
            - `weight_decay` (float): Weight decay (L2 penalty).

    Returns:
        torch.optim.Optimizer: Instantiated PyTorch optimizer.
    """
    return torch.optim.Adam(
        params,
        lr=config["lr"],
        betas=config["betas"],
        eps=config["eps"],
        weight_decay=config["weight_decay"],
    )
