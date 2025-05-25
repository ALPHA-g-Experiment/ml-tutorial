#!/usr/bin/env python3

import argparse
import numpy as np
import polars as pl
import torch

parser = argparse.ArgumentParser(
    description="Test a model", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("model", help="path to serialized model (.pt)")
parser.add_argument("dataset", help="path to Parquet dataset")
parser.add_argument("--output", help="write output to `OUTPUT`")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.jit.load(args.model, map_location=device)
model.eval()

targets = []
predictions = []
with torch.no_grad():
    for entry in pl.read_parquet(args.dataset).iter_rows(named=True):
        target = torch.tensor(entry["target"][2], device=device).unsqueeze(0)
        point_cloud = (
            torch.tensor(entry["point_cloud"], device=device)
            .transpose(0, 1)
            .unsqueeze(0)
        )

        prediction = model(point_cloud)

        targets = np.append(targets, target.cpu().numpy())
        predictions = np.append(predictions, prediction.cpu().numpy())


df = pl.DataFrame({"target": targets, "prediction": predictions})
if args.output:
    df.write_csv(args.output)
    print(f"Created `{args.output}`")
else:
    print(df.write_csv())
