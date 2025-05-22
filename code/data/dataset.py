import polars as pl
from torch.utils.data import Dataset


class PointCloudDataset(Dataset):
    """
    A PyTorch dataset for point cloud data stored in a Parquet file.

    Args:
        parquet_file (str): Path to the Parquet file containing samples.
            Each row must include:
                - `target`: Annihilation vertex position (array[f32, 3])
                - `point_cloud`: list of 3D points (list[array[f32, 3]])
        config (dict): Dataset configuration with keys:
            - `cloud_size` (int): Number of points to sample from the point cloud.

    Returns:
        A dataset compatible with PyTorch where each item is a tuple:
            (Tensor of shape (3, cloud_size), scalar Tensor)
    """

    def __init__(self, parquet_file, config):
        cloud_size = config["cloud_size"]

        self.inner = (
            pl.scan_parquet(parquet_file)
            .with_columns(
                pl.col("target").arr.get(2),
                pl.col("point_cloud")
                .list.eval(
                    pl.element().sort_by(
                        pl.element().arr.get(0).pow(2) + pl.element().arr.get(1).pow(2)
                    )
                )
                .list.gather(
                    pl.int_ranges(cloud_size) % pl.col("point_cloud").list.len()
                )
                .list.to_array(cloud_size)
                # PyTorch pipeline expects (B, C, L) shape
                .map_batches(lambda x: x.to_numpy().transpose(0, 2, 1)),
            )
            .collect()
            .to_torch("dataset", label="target")
        )

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        return self.inner[idx]
