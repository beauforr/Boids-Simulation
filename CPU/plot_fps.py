#!/usr/bin/env python3

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import math

def read_csv_with_pandas(path: Path):
    import pandas as pd
    df = pd.read_csv(path)
    # Normalize column names by stripping whitespace
    df.columns = [str(c).strip() for c in df.columns]
    return df

def read_csv_fallback(path: Path):
    import csv
    rows = []
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(r)
    # convert to dict-of-lists with stripped header names
    cols = {}
    for r in rows:
        for k, v in r.items():
            key = str(k).strip() if k is not None else k
            cols.setdefault(key, []).append(v)
    return cols

def detect_and_prepare(data, path: Path):
    # helper for name normalization
    def find_col_case_insensitive(cols, target):
        for c in cols:
            if str(c).strip().lower() == target:
                return c
        return None

    series = {}

    try:
        import pandas as pd  # type: ignore
        is_df = isinstance(data, pd.DataFrame)
    except Exception:
        is_df = False

    if is_df:
        df = data
        cols = list(df.columns)

        boids_col = find_col_case_insensitive(cols, 'boids')
        if boids_col is None:
            raise SystemExit(f"Could not find a 'boids' column in {path}. Found: {cols}")

        naive_col = find_col_case_insensitive(cols, 'naive')
        grid_col = find_col_case_insensitive(cols, 'grid')

        if not naive_col and not grid_col:
            raise SystemExit(f"Could not find 'naive' or 'grid' columns in {path}. Found: {cols}")

        x = df[boids_col].astype(int).tolist()
        if naive_col:
            series['naive'] = (x, df[naive_col].astype(float).tolist())
        if grid_col:
            series['grid'] = (x, df[grid_col].astype(float).tolist())

        return series, boids_col

    else:
        cols = list(data.keys())

        boids_col = find_col_case_insensitive(cols, 'boids')
        if boids_col is None:
            raise SystemExit(f"Could not find a 'boids' column in {path}. Found: {cols}")

        naive_col = find_col_case_insensitive(cols, 'naive')
        grid_col = find_col_case_insensitive(cols, 'grid')

        if not naive_col and not grid_col:
            raise SystemExit(f"Could not find 'naive' or 'grid' columns in {path}. Found: {cols}")

        # convert lists to proper types, stripping values first
        x = [int(str(v).strip()) for v in data[boids_col]]
        if naive_col:
            series['naive'] = (x, [float(str(v).strip()) for v in data[naive_col]])
        if grid_col:
            series['grid'] = (x, [float(str(v).strip()) for v in data[grid_col]])

        return series, boids_col

def plot_series(series: dict, x_label: str, out_path: Path, show: bool, title: str | None, x_log: bool):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    colors = {'naive': 'tab:blue', 'grid': 'tab:orange'}
    for name, (xs, ys) in series.items():
        # sort pairs by x
        pairs = sorted(zip(xs, ys), key=lambda p: p[0])
        xs_s, ys_s = zip(*pairs)
        plt.plot(xs_s, ys_s, marker='o', label=str(name), color=colors.get(name, None))

    plt.xlabel('Number of boids (N)')
    plt.xscale('log')
    plt.ylabel('Frames per second (FPS)')
    plt.yscale('log')
    plt.title('Boids CPU Simulation: FPS vs Number of Boids')
    plt.legend()
    plt.grid(True, which='both', ls='--', lw=0.5)
    if x_log:
        plt.xscale('log')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")
    if show:
        plt.show()

def main():
    p = argparse.ArgumentParser(description='Plot FPS vs Boids for CPU methods (naive vs grid)')
    p.add_argument('--input', '-i', type=Path, default=Path('CPU/fps.csv'), help='Input CSV file')
    p.add_argument('--output', '-o', type=Path, default=Path('CPU/fps_plot.png'), help='Output image file')
    p.add_argument('--show', action='store_true', help='Show the plot (blocks)')
    p.add_argument('--title', type=str, default=None, help='Optional plot title')
    p.add_argument('--xlog', action='store_true', help='Use log scale for x-axis')
    args = p.parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}")
        sys.exit(2)

    # try pandas first
    try:
        data = read_csv_with_pandas(args.input)
    except Exception:
        data = read_csv_fallback(args.input)

    series, x_label = detect_and_prepare(data, args.input)
    if not series:
        print('No series detected in file')
        sys.exit(3)

    plot_series(series, x_label, args.output, args.show, args.title, args.xlog)

if __name__ == '__main__':
    main()
