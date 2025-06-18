# Data Preprocessing

Before we can train a model, we need to split our dataset into three parts:
- Training set: The data the model learns from.
- Validation set: Used to tune hyperparameters and monitor performance during
  training.
- Test set: Used only at the end to evaluate final model performance.

Create a new script in the `ml-tutorial/code` directory. You can then create
these splits like:

```python
import polars as pl

complete_df = pl.read_parquet("../data/raw_data.parquet")

# Example: Take the first 10000 events
subset_df = complete_df.slice(offset=0, length=10000)
subset_df.write_parquet("../data/first_10000.parquet")
```

> **Activity:**  
> - Create a training subset with 80% of the data.
> - Create a validation subset with 10% of the data.
> - Create a test subset with the remaining 10% of the data.
>
> Note that these splits should not overlap. Shared events between the splits
> can lead to overfitting and an overly optimistic evaluation of the model's
> performance.

Once you've created your splits, it's a good idea to repeat the **data
exploration steps** on each one of them. Check the target distribution and point
cloud sizes to make sure nothing looks unusual (which could affect the model's
performance if the splits are not representative of the whole dataset).

> Tip: If you want to shuffle a dataset, you can do it with:
>
> ```python
> shuffled_df = df.sample(fraction=1.0, shuffle=True)
> ```
