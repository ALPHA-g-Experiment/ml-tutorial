# Training the Baseline Model

Now that your data is prepared, let's train a model that predicts the vertex
position from detector hits.

We've provided a 
[reference model](https://github.com/ALPHA-g-Experiment/ml-tutorial/blob/main/code/model/regressor.py)
and
[training script](https://github.com/ALPHA-g-Experiment/ml-tutorial/blob/main/code/train.py),
which you can run like:

```bash
python train.py /path/to/train /path/to/validate --output-dir /path/to/output
```
