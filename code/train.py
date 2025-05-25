#!/usr/bin/env python3

import argparse
import csv
import torch

from data.dataset import PointCloudDataset
from datetime import datetime
from dynaconf import loaders
from dynaconf.utils.boxing import DynaBox
from model.regressor import Regressor
from pathlib import Path
from config.settings import settings
from training.optimizer import build_optimizer
from training.loop import train_one_epoch, test_one_epoch
from training.loss import CustomLoss
from torch.utils.data import DataLoader

# Marginally faster, and no noticeable difference in accuracy. Silences PyTorch
# suggestions to use "high" precision for matmul in a100.
torch.set_float32_matmul_precision("high")


parser = argparse.ArgumentParser(description="Training pipeline")
parser.add_argument("train_data", help="path to Parquet training data")
parser.add_argument("val_data", help="path to Parquet validation data")
parser.add_argument(
    "--output-dir", type=Path, help="output directory (default: YYYY_MM_DDTHH-MM-SS)"
)
parser.add_argument(
    "--force", action="store_true", help="overwrite OUTPUT_DIR if it already exists"
)
parser.add_argument("--dry-run", action="store_true", help="only output settings file")

args = parser.parse_args()
if args.output_dir is None:
    args.output_dir = Path(datetime.now().strftime("%Y_%m_%dT%H-%M-%S"))

if args.output_dir.exists() and not args.force:
    raise FileExistsError(
        f"Output directory `{args.output_dir}` already exists. Use --force to overwrite."
    )
args.output_dir.mkdir(parents=True, exist_ok=True)

loaders.write(str(args.output_dir / "config.toml"), DynaBox(settings.to_dict()))

if args.dry_run:
    exit(0)

batch_size = settings.training.batch_size
max_epochs = settings.training.max_epochs
train_dataset = PointCloudDataset(args.train_data, settings.data)
validation_dataset = PointCloudDataset(args.val_data, settings.data)
model = Regressor(settings.model)
loss_fn = CustomLoss(settings.training.loss)
optimizer = build_optimizer(model.parameters(), settings.training.optimizer)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
)
validation_dataloader = DataLoader(
    validation_dataset, batch_size=batch_size, pin_memory=True
)

training_log = args.output_dir / "training_log.csv"
with open(training_log, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "training_loss", "validation_loss", "mean", "std"])

best_loss = float("inf")
for i in range(max_epochs):
    train_loss = train_one_epoch(train_dataloader, model, loss_fn, optimizer, device)
    validation_loss, mean, std = test_one_epoch(
        validation_dataloader, model, loss_fn, device
    )

    with open(training_log, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([i, train_loss, validation_loss, mean, std])

    if validation_loss < best_loss:
        best_loss = validation_loss

        model_scripted = torch.jit.script(model)
        model_scripted.save(args.output_dir / "model.pt")
