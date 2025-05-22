#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import polars as pl


def target_z(dataset, bins, min, max, normalize, alpha):
    z = pl.scan_parquet(dataset).select(z=pl.col("target").arr.get(2)).collect()["z"]
    min = z.min() if min is None else min
    max = z.max() if max is None else max

    plt.hist(
        z, bins=bins, range=(min, max), density=normalize, alpha=alpha, label=dataset
    )


def cloud_size(dataset, bins, min, max, normalize, alpha):
    size = (
        pl.scan_parquet(dataset)
        .select(size=pl.col("point_cloud").list.len())
        .collect()["size"]
    )
    min = size.min() if min is None else min
    max = size.max() if max is None else max

    plt.hist(
        size, bins=bins, range=(min, max), density=normalize, alpha=alpha, label=dataset
    )


parser = argparse.ArgumentParser(description="Data Visualization")
subparsers = parser.add_subparsers(dest="subcommand", required=True)

parser_all = argparse.ArgumentParser(add_help=False)
parser_all.add_argument(
    "data",
    nargs="+",
    help="path to a data file (expected format depends on the subcommand)",
)
parser_all.add_argument("--output", help="write output to `OUTPUT`")

parser_hist = argparse.ArgumentParser(add_help=False)
parser_hist.add_argument("--bins", type=int, default=100, help="number of bins")
parser_hist.add_argument("--max", type=float, help="maximum value")
parser_hist.add_argument("--min", type=float, help="minimum value")
parser_hist.add_argument("--normalize", action="store_true", help="normalize data")

parser_target_z = subparsers.add_parser(
    "target-z",
    help="target z distribution (expects Parquet datasets)",
    parents=[parser_all, parser_hist],
)

parser_cloud_size = subparsers.add_parser(
    "cloud-size",
    help="point cloud size distribution (expects Parquet datasets)",
    parents=[parser_all, parser_hist],
)

args = parser.parse_args()

alpha = 1.0 if len(args.data) == 1 else 0.5
for dataset in args.data:
    match args.subcommand:
        case "target-z":
            target_z(dataset, args.bins, args.min, args.max, args.normalize, alpha)
            plt.xlabel("z [mm]")
            plt.ylabel("Count")

        case "cloud-size":
            cloud_size(dataset, args.bins, args.min, args.max, args.normalize, alpha)
            plt.xlabel("Number of points")
            plt.ylabel("Count")

plt.legend()

if args.output:
    plt.savefig(args.output)
    print(f"Created `{args.output}`")
else:
    plt.show()
