# Hyperparameter Tuning

Now that you have made some improvements to the model, it's time to focus on
**hyperparameter tuning**. Small changes in settings like batch size, cloud
size, or number of epochs can significantly impact model performance.

The provided training script supports setting hyperparameters using
**environment variables**, making it easy to scan different values using e.g.
simple shell scripts. For example, you can set the number of epochs like this:

```bash
RECO_TRAIN_TRAINING__MAX_EPOCHS=100 python train.py /path/to/train /path/to/val
```

> **Activity:**  
> Set up a hyperparameter scan to run overnight. You can focus on as many
> hyperparameters as you like.

Note that you can use the `--dry-run` flag to test your hyperparameter
assignments. This will only save the used values to a file without actually
running the training.

## Supported Hyperparameters

Here's a summary of tunable hyperparameters and how to set them:

| Environment Variable | Hyperparameter&nbsp;Description | Default Value |
|----------------------|-------------|---------------|
| `RECO_TRAIN_DATA__CLOUD_SIZE` | Size of the point cloud | 140 |
| `RECO_TRAIN_MODEL__CONV_FEATURE_EXTRACTOR_PRE` | List of Conv1d layer dimensions for pre-alignment feature extraction | [64] |
| `RECO_TRAIN_MODEL__CONV_FEATURE_EXTRACTOR_POST` | List of Conv1d layer dimensions for post-alignment feature extraction | [128,1024] |
| `RECO_TRAIN_MODEL__FC_REGRESSOR` | List of Linear layer dimensions | [512,256] |
| `RECO_TRAIN_MODEL__INPUT_TRANSFORM_NET__CONV_FEATURE_EXTRACTOR` | Inner `_TNet` (z-shift) list of Conv1d layer dimensions | [64,128,1024] |
| `RECO_TRAIN_MODEL__INPUT_TRANSFORM_NET__FC_REGRESSOR` | Inner `_TNet` (z-shift) list of Linear layer dimensions | [512,256] |
| `RECO_TRAIN_MODEL__FEATURE_TRANSFORM_NET__CONV_FEATURE_EXTRACTOR` | Inner `_TNet` (feature alignment) list of Conv1d layer dimensions | [64,128,1024] |
| `RECO_TRAIN_MODEL__FEATURE_TRANSFORM_NET__FC_REGRESSOR` | Inner `_TNet` (feature alignment) list of Linear layer dimensions | [512,256] |
| `RECO_TRAIN_TRAINING__BATCH_SIZE` | Batch size | 64 |
| `RECO_TRAIN_TRAINING__MAX_EPOCHS` | Maximum number of training epochs | 50 |

Note you can mix and match any of these. Just prefix your training command with
the relevant environment variable assignments.
