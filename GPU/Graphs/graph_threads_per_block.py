import numpy as np
import matplotlib.pyplot as plt

# Read data from measurements.csv
from pathlib import Path
csv_path = Path(__file__).resolve().parent / 'threads_per_block.csv'
if not csv_path.exists():
	raise FileNotFoundError(f"threads_per_block.csv not found at {csv_path}")
data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)

threads_per_block = data[:, 0]
fps = data[:, 1]

plt.figure(figsize=(10, 6))

plt.plot(threads_per_block, fps, marker='o', label='FPS', linewidth=2, color='blue')

plt.title("Effect of Threads per Block on FPS (N=10,000)")
plt.xlabel("Threads per block")
plt.ylabel("Frames per second (FPS)")
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()

plt.show()
