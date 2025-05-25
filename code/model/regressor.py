import torch
import torch.nn as nn


class _TNet(nn.Module):
    """
    A network that learns an alignment transformation.

    Args:
        in_dim (int): Input dimension.
        config (dict): Inner layer configuration with keys:
            - `conv_feature_extractor`: list of Conv1d layer dimensions.
            - `fc_regressor`: list of Linear layer dimensions.
        out_dim (int): Output dimension.

    Inputs:
        x (Tensor): Input tensor of shape (B, C, L), where:
            B = batch size,
            C = number of input features (in_dim),
            L = number of points.

    Returns:
        Tensor: Transformation coefficients of shape (B, out_dim).
    """

    def __init__(self, in_dim, config, out_dim):
        super().__init__()

        self.conv_feature_extractor = nn.Sequential()
        for dim in config["conv_feature_extractor"]:
            self.conv_feature_extractor.append(nn.Conv1d(in_dim, dim, 1))
            self.conv_feature_extractor.append(nn.BatchNorm1d(dim))
            self.conv_feature_extractor.append(nn.ReLU())
            in_dim = dim

        self.fc_regressor = nn.Sequential()
        for dim in config["fc_regressor"]:
            self.fc_regressor.append(nn.Linear(in_dim, dim))
            self.fc_regressor.append(nn.BatchNorm1d(dim))
            self.fc_regressor.append(nn.ReLU())
            in_dim = dim

        # Keep the last layer separate just to make it easier to e.g. initialize
        # it in Regressor's feature alignment T-Net
        self.last = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.conv_feature_extractor(x)
        x = x.max(dim=2).values
        x = self.fc_regressor(x)
        x = self.last(x)
        return x


class Regressor(nn.Module):
    """
    Point cloud regression network based on a PointNet-style architecture.

    The model extracts point features, applies a learned feature alignment
    transformation, aggregates global features, and predicts a single scalar
    value (annihilation vertex z-coordinate).

    Args:
        config (dict): Inner layer configuration with keys:
            - `input_transform_net`: nested config for input alignment T-Net.
            - `conv_feature_extractor_pre`: list of Conv1d layer dimensions for
              pre-alignment feature extraction.
            - `feature_transform_net`: nested config for feature alignment T-Net.
            - `conv_feature_extractor_post`: list of Conv1d layer dimensions for
              post-alignment feature extraction.
            - `fc_regressor`: list of Linear layer dimensions.

    Inputs:
        x (Tensor): Input tensor of shape (B, 3, L), where:
            B = batch size,
            3 = number of input features (x, y, z coordinates),
            L = number of points.

    Returns:
        Tensor: Predicted scalar value of shape (B,).
    """

    def __init__(self, config):
        super().__init__()
        in_dim = 3

        self.conv_feature_extractor_pre = nn.Sequential()
        for dim in config["conv_feature_extractor_pre"]:
            self.conv_feature_extractor_pre.append(nn.Conv1d(in_dim, dim, 1))
            self.conv_feature_extractor_pre.append(nn.BatchNorm1d(dim))
            self.conv_feature_extractor_pre.append(nn.ReLU())
            in_dim = dim

        self.num_feats = in_dim
        self.feature_transform_net = _TNet(
            in_dim, config["feature_transform_net"], in_dim * in_dim
        )
        with torch.no_grad():
            self.feature_transform_net.last.bias += torch.eye(in_dim).flatten()

        self.conv_feature_extractor_post = nn.Sequential()
        for dim in config["conv_feature_extractor_post"]:
            self.conv_feature_extractor_post.append(nn.Conv1d(in_dim, dim, 1))
            self.conv_feature_extractor_post.append(nn.BatchNorm1d(dim))
            self.conv_feature_extractor_post.append(nn.ReLU())
            in_dim = dim

        self.fc_regressor = nn.Sequential()
        for dim in config["fc_regressor"]:
            self.fc_regressor.append(nn.Linear(in_dim, dim))
            self.fc_regressor.append(nn.BatchNorm1d(dim))
            self.fc_regressor.append(nn.ReLU())
            in_dim = dim

        self.fc_regressor.append(nn.Linear(in_dim, 1))

    def forward(self, x):
        x = self.conv_feature_extractor_pre(x)

        feat_trans = self.feature_transform_net(x).view(
            -1, self.num_feats, self.num_feats
        )
        x = feat_trans.bmm(x)

        x = self.conv_feature_extractor_post(x)
        x = x.max(dim=2).values
        x = self.fc_regressor(x)

        return x.view(-1)
