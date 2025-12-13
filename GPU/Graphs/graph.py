import numpy as np
import matplotlib.pyplot as plt

# Read data from measurements.csv
from pathlib import Path
csv_path = Path(__file__).resolve().parent / 'measurements.csv'
if not csv_path.exists():
	raise FileNotFoundError(f"measurements.csv not found at {csv_path}")
data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)

N_values = data[:, 0]
fps_cpu = data[:, 1]
fps_gpu = data[:, 2]
fps_gpu_improved = data[:, 3]
fps_gpu_grid = data[:, 4]
fps_cpu_grid = data[:, 5]

# Replace the simple plt.figure/plt.plot block with a figure + twin axis to plot speedup
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(N_values, fps_cpu, marker='o', label='CPU (naive)', linewidth=2, color='blue')
ax.plot(N_values, fps_gpu, marker='o', label='GPU (naive)', linewidth=2, color='limegreen')
ax.plot(N_values, fps_gpu_improved, marker='o', label='GPU (naive improved)', linewidth=2, color='green')
ax.plot(N_values, fps_gpu_grid, marker='o', label='GPU (grid)', linewidth=2, color='darkgreen')
ax.plot(N_values, fps_cpu_grid, marker='o', label='CPU (grid)', linewidth=2, color='navy')

ax.set_title("Boids simulation: FPS comparison CPU vs GPU")
ax.set_xlabel("Number of boids (N)")
ax.set_ylabel("Frames per second (FPS)")
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, linestyle='--', alpha=0.4)


ax.legend(loc='best')

plt.tight_layout()
plt.show()
