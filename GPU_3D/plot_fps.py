#!/usr/bin/env python3
"""Simple CSV plotter for columns `N` and `fps`.

Usage:
  python plot_fps.py [csv_file] [--out out.png] [--noshow]

Defaults to `measurements.csv` in the same folder.
"""
import argparse
import csv
import os
import sys
import matplotlib.pyplot as plt


def read_csv(path):
    Ns = []
    fps = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"File '{path}' appears empty or has no header")
        if 'N' not in reader.fieldnames or 'fps' not in reader.fieldnames:
            raise SystemExit(f"CSV must contain columns 'N' and 'fps'. Found: {reader.fieldnames}")
        for row in reader:
            try:
                nval = int(row['N'])
            except Exception:
                try:
                    nval = float(row['N'])
                except Exception:
                    continue
            try:
                fval = float(row['fps'])
            except Exception:
                fval = float('nan')
            Ns.append(nval)
            fps.append(fval)
    return Ns, fps


def plot_data(Ns, fps, out=None, show=True):
    # Sort by N for a clean line plot
    pairs = sorted(zip(Ns, fps), key=lambda x: x[0])
    Ns_sorted, fps_sorted = zip(*pairs)

    plt.figure(figsize=(8, 5))
    plt.plot(Ns_sorted, fps_sorted, marker='o', linestyle='-')
    plt.scatter(Ns_sorted, fps_sorted, s=30)
    plt.xlabel('Number of Foids (N)')
    plt.ylabel('Frames per second (FPS)')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('3D Foids simulation: FPS vs N')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if out:
        plt.savefig(out, dpi=150)
        print(f"Saved plot to: {out}")
    if show:
        plt.show()



def main():
    # Default to measurements.csv located next to this script (so it works when run from repo root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(script_dir, 'measurements.csv')

    parser = argparse.ArgumentParser(description='Plot N vs FPS from CSV file')
    parser.add_argument('csv', nargs='?', default=default_csv, help=f'CSV file to read (default: {default_csv})')
    parser.add_argument('--out', '-o', help='Write plot to PNG file')
    parser.add_argument('--noshow', action='store_true', help='Do not open an interactive window')
    args = parser.parse_args()

    try:
        Ns, fps = read_csv(args.csv)
    except FileNotFoundError:
        print(f"Error: file not found: {args.csv}")
        sys.exit(2)
    except Exception as e:
        print('Error:', e)
        sys.exit(1)

    if not Ns:
        print('No data found in', args.csv)
        sys.exit(1)

    print(f"Using CSV: {args.csv}")

    plot_data(Ns, fps, out='3D-simulation.png', show=not args.noshow)


if __name__ == '__main__':
    main()
