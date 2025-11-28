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

plt.figure(figsize=(10, 6))

plt.plot(N_values, fps_cpu, marker='o', label='CPU', linewidth=2, color='blue')
plt.plot(N_values, fps_gpu, marker='o', label='GPU (256thr/blck)', linewidth=2, color='limegreen')
plt.plot(N_values, fps_gpu_improved, marker='o', label='GPU (128thr/blck)', linewidth=2, color='green')

plt.title("Boids simulation: FPS comparison CPU vs GPU")
plt.xlabel("Number of boids (N)")
plt.ylabel("Frames per second (FPS)")
plt.xscale('log')
plt.yscale('log')
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()

plt.show()
