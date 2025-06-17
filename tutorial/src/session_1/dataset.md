# Exploring the Data

Each annihilation event produces a number of charged particles that leave a
trail of hits in the ALPHA-g detector.

Each event in the dataset (`ml-tutorial/data/raw_data.parquet`) contains:
- A true vertex position: `(x, y, z)` i.e. the origin of the annihilation.
- A set of 3D hit positions: `[(x1, y1, z1), ..., (xn, yn, zn)]`.

```python
import polars as pl

df = pl.read_parquet("/path/to/ml_tutorial/data/raw_data.parquet")
print(df)

"""
shape: (100_000, 2)
┌─────────────────────────────────┬─────────────────────────────────┐
│ target                          ┆ point_cloud                     │
│ ---                             ┆ ---                             │
│ array[f32, 3]                   ┆ list[array[f32, 3]]             │
╞═════════════════════════════════╪═════════════════════════════════╡
│ [1.455413, 15.901725, -571.578… ┆ [[46.253021, 175.486893, -558.… │
│ [22.550814, 3.005712, 834.1053… ┆ [[171.600311, -59.034386, 1070… │
│ [4.511479, -8.75235, -1014.025… ┆ [[33.221375, -178.431686, -106… │
│ [-9.729183, 4.537313, -970.073… ┆ [[33.232174, -178.489685, -110… │
│ [16.184988, -9.351818, -66.521… ┆ [[152.121017, -98.965576, -218… │
│ …                               ┆ …                               │
│ [5.014961, -12.403949, 62.5900… ┆ [[-158.989197, -87.506714, 226… │
│ [-7.504006, -18.486027, 689.96… ┆ [[-181.138474, 11.128488, -30.… │
│ [-19.474358, 13.368332, -59.90… ┆ [[-120.215218, -135.953278, -1… │
│ [-8.580782, 1.076302, -866.250… ┆ [[15.571182, -180.818787, -846… │
│ [-18.750277, 6.330079, 969.173… ┆ [[176.59346, -41.938202, 798.0… │
└─────────────────────────────────┴─────────────────────────────────┘
"""
```

Before training any model, it's important to **understand the structure and
characteristics of the data**.

> **Activity:**  
> - Where do annihilation events occur? A skewed distribution in vertex z
>   positions might cause the model to "cheat" by always guessing the most
>   common region.
> - Do all events have the same number of hits? Variable-length point clouds
>   will require special handling in the model architecture.

To help you answer these questions, we've provided the script:
`ml-tutorial/code/visualization.py`.

You can run it directly to visualize key properties of a dataset:
```bash
# Target z distribution
python visualize.py target-z /path/to/dataset.parquet

# Point cloud size distribution
python visualize.py cloud-size /path/to/dataset.parquet
```

## Iterating Through the Dataset with PyTorch

To train a model, we need to iterate through the dataset. PyTorch provides a
primitive
[`torch.utils.data.Dataset`](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
class that allows us to decouple the data loading from the model
training/batching process.

We've provided a
[PyTorch-compatible dataset class](https://github.com/ALPHA-g-Experiment/ml-tutorial/blob/main/code/data/dataset.py).
It wraps a `.parquet` file and gives you easy access to the data in a
PyTorch-friendly way:

```python
from data.dataset import PointCloudDataset

config = {"cloud_size": 140}
dataset = PointCloudDataset("/path/to/dataset.parquet", config)

index = 0 # First event
point_cloud, target = dataset[index]
```

Try running the code above and plot some point clouds and their corresponding
targets (annihilation vertices).

> **Activity:**  
> - Inspect the
>   [`PointCloudDataset`](https://github.com/ALPHA-g-Experiment/ml-tutorial/blob/main/code/data/dataset.py#L5-L50)
>   class. How does it handle variable-length point clouds?
> - Using the first 10 events (indices 0-9), plot the point clouds and their
>   targets. Do they look like you expected?  
>   You can make a 3D scatter plot using `matplotlib`:
>   ```python
>   import matplotlib.pyplot as plt
>
>   fig = plt.figure()
>   ax = fig.add_subplot(projection="3d")
>
>   ax.scatter(point_cloud[0], point_cloud[1], point_cloud[2])
>   ax.scatter(0, 0, target.item(), color="red")
>   ```
> - Using the next 10 events (indices 10-19), plot the point clouds without
>   their targets. Make an educated guess about the target vertex position
>   based on the point cloud. Compare your guess with the actual target
>   positions.
