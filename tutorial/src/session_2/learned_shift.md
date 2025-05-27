# Improving the Model

Now that you understand the full structure of the model, it's time to explore a
simple but effective improvement.

As you saw in the previous section, the model has an
[inner `_TNet`](https://github.com/ALPHA-g-Experiment/ml-tutorial/blob/e23bb159fc8c5bc7782a5474e89dc542403a7740/code/model/regressor.py#L95-L97)
that learns the coefficients of a matrix used to
[align the abstract features](https://github.com/ALPHA-g-Experiment/ml-tutorial/blob/e23bb159fc8c5bc7782a5474e89dc542403a7740/code/model/regressor.py#L120-L123).
This **feature alignment** allows the network to adaptively transform the point
features, often leading to better performance.

## Our New Idea: A Learnable Z-Shift

We're now going to add a **second internal** `_TNet`, but this one has a
different goal: Instead of predicting a full matrix, it will predict a single
scalar; a **z-shift**.

This shift will be subtracted from the z-coordinate of all points before any
feature extraction. The idea is to let the network:

1. First guess a coarse global z-position of the vertex, and
2. Then predict the small **delta** from this shifted position.

> **Activity:**  
> - Modify your `Regressor` class to include a new `_TNet` instance at the start
>   of your model (this `_TNet` should have an `out_dim` of 1).
> - Use this new `_TNet` to predict a z-shift given the input point cloud
>   (before any feature extraction).
> - Subtract this z-shift from the z-coordinate of all points in the point
>   cloud. You can do this like:
>   ```python
>   x = torch.stack((x[:, 0, :], x[:, 1, :], x[:, 2, :] - input_trans), dim=1)
>   ```
> - Let the rest of the model continue unchanged i.e. it is now predicting a
>   delta from the shifted point cloud.
> - Update the return value of the `forward` method to include the z-shift plus
>   the delta.

Once you've made these changes, retrain the model and evaluate it again. Do you
see an improvement in performance?
