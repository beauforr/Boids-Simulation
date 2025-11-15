import numpy as np
import matplotlib.pyplot as plt

# amoount of boids
N_values = np.array([100, 300, 500, 1000, 2000, 5000, 10000, 20000, 50000])

# fps values
fps_cpu = np.array([400, 140, 75, 35, 12, 5, 1.6, 0.4, 0])
fps_gpu = np.array([250, 240, 230, 210, 180, 140, 100, 60, 30]) 
plt.figure(figsize=(10, 6))

plt.plot(N_values, fps_cpu, marker='o', label='CPU (dummy)', linewidth=2)
plt.plot(N_values, fps_gpu, marker='o', label='GPU (dummy)', linewidth=2)

plt.title("Boids simulation: FPS comparison CPU vs GPU")
plt.xlabel("Number of boids (N)")
plt.ylabel("Frames per second (FPS)")
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()

plt.show()
