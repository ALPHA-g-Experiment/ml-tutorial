# Training and Evaluating the Baseline Model

Now that your data is prepared, let's train a model that predicts the vertex
position from detector hits.

We've provided a 
[reference model](https://github.com/ALPHA-g-Experiment/ml-tutorial/blob/main/code/model/regressor.py)
and
[training script](https://github.com/ALPHA-g-Experiment/ml-tutorial/blob/main/code/train.py),
which you can run like:

```bash
python train.py /path/to/train.parquet /path/to/validate.parquet --output-dir /path/to/output
```

This will run a full training loop and create the following files in the
`/path/to/output` directory:
- `config.toml`: the hyperparameters used during training.
- `training_log.csv`: a CSV file with the training and validation loss at each
  epoch.
- `model.pth`: the trained model.

Training the model can take a while depending on the size of your dataset, the
number of epochs, and the hardware you are using. By default, the training
script will run for 50 epochs. You can inspect the `training_log.csv` file to
monitor training progress.

> **Activity:**  
> Write a script to plot the training and validation loss as a function of
> epoch.  
> What do you observe?

Once the model is trained, you can evaluate how well it performs on the
validation set using the provided
[test script](https://github.com/ALPHA-g-Experiment/ml-tutorial/blob/main/code/test.py):

```bash
python test.py /path/to/model.pt /path/to/validate.parquet --output /path/to/output.csv
```

This will create a CSV file with both the true and predicted vertex positions
for each event.

> **Activity:**  
> - Write a script to plot predicted vertex position vs. true vertex position.
> - Create a histogram of the residuals (predicted - true).
> - Plot the residuals as a function of true z-position. Is there any pattern?
