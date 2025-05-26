# Understanding the Model Architecture

Before improving our model, let's understand its structure.

The model you're using is a simplified
[PointNet](https://arxiv.org/abs/1612.00593). Surprisingly, we can understand
everything by just studying the very simple (~20 actual lines of code)
[`_TNet`](https://github.com/ALPHA-g-Experiment/ml-tutorial/blob/08fee4cec46e84840a2f9ddd32a5ff9ef48cd8b4/code/model/regressor.py#L5-L52)
class.

This architecture consists of three main components:
1. **Feature Extraction**:  
    The input tensor has shape `(B, C, L)`, where `B` is the batch size, `C` is
    the number of channels (input features), and `L` is the number of points.

    The [feature extractor](https://github.com/ALPHA-g-Experiment/ml-tutorial/blob/08fee4cec46e84840a2f9ddd32a5ff9ef48cd8b4/code/model/regressor.py#L29-L33)
    applies a series of [1D convolution](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html)
    layers to represent the input points in a higher-dimensional space of
    abstract features.

    Let's see a minimal example:

    ```python
    import torch
    import torch.nn as nn

    x = torch.randn(2, 3, 4)
    print("Input shape:", x.shape)

    net = nn.Conv1d(3, 5, 1)
    x = net(x)
    print("Output shape:", x.shape)
    ```

    Note that the feature extractor also includes
    [batch normalization](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)
    and
    [ReLU activation](https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html)
    layers, which are essential for training deep neural networks.

    > **Activity**:  
    > Change the `net` in the example above to `nn.BatchNorm1d(3)` or
    > `nn.ReLU()`, and manually compute what these layers would do to the input
    > tensor (print `x` before and after the operation to verify your
    > calculations).

    Together, these layers make up the entire feature extractor. Just basic
    operations, chained one after another. Once you break it down, there's no
    mystery!
