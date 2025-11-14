"""
GPU boids simulation (PyCUDA) + matplotlib visualization.

This script runs the boids update kernel on the GPU (naive O(N^2) neighbor checks)
and visualizes positions in Python using Matplotlib. It's written for clarity and
as a stepping stone to more advanced GPU implementations (spatial hashing, shared
memory tiling, etc.).

Usage (from `GPU/`):
  source .venv/bin/activate   # activate your venv if needed
  python main.py

Notes:
- This uses PyCUDA (pycuda) and requires a working CUDA toolchain and drivers.
- The kernel is simple and intentionally mirrors the CPU logic for clarity.
"""

import math
import numpy as np
import time
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------- Simulation parameters (mirror CPU defaults) ----------
W, H = 8000, 5000
N = 20000                      # number of boids (tune for your GPU)
visual_range = 40.0
protected_range = 8.0
centering_factor = 0.0005
avoid_factor = 0.05
matching_factor = 0.05
turn_factor = 0.2
margin = 100.0
min_speed = 3.0
max_speed = 6.0
dt = 1.0

fraction_biased = 0.05
max_bias = 0.01
default_bias = 0.001

# ---------- Helpers / initialization ----------
def create_boids(num):
  num = int(max(1, round(num)))
  pos = np.random.rand(num, 2).astype(np.float32) * np.array([W, H], dtype=np.float32)
  angles = np.random.rand(num).astype(np.float32) * 2.0 * np.pi
  vel = np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float32)
  v0 = (min_speed + max_speed) / 2.0
  vel *= np.float32(v0)
  bias_group = np.zeros(num, dtype=np.int32)
  bias_val = np.full(num, default_bias, dtype=np.float32)
  # randomize biased groups
  num_biased = int(num * fraction_biased)
  if num_biased > 0:
    inds = np.random.choice(num, num_biased, replace=False)
    half = num_biased // 2
    bias_group[inds[:half]] = 1
    bias_group[inds[half:]] = -1
    if max_bias > default_bias:
      vals = np.random.rand(num_biased).astype(np.float32) * (max_bias - default_bias) + default_bias
      bias_val[inds] = vals
  return pos, vel, bias_group, bias_val


# ---------- CUDA kernel (naive O(N^2) neighbor checks) ----------
# Keep kernel readable - it mirrors the CPU update logic.
kernel_code = r"""
#include <math.h>
extern "C" {
__global__ void update_boids(
  float *pos_x, float *pos_y,
  float *vel_x, float *vel_y,
  int *bias_group, float *bias_val,
  int N,
  float visual_range, float protected_range,
  float centering_factor, float avoid_factor, float matching_factor,
  float turn_factor, float margin, float min_speed, float max_speed, float dt,
  int W, int H)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  float xi = pos_x[i];
  float yi = pos_y[i];
  float new_vx = vel_x[i];
  float new_vy = vel_y[i];

  float close_dx = 0.0f;
  float close_dy = 0.0f;
  float xavg = 0.0f;
  float yavg = 0.0f;
  float vxavg = 0.0f;
  float vyavg = 0.0f;
  int neighbors = 0;

  float vr = visual_range;
  float pr = protected_range;
  float pr2 = pr * pr;
  float vr2 = vr * vr;

  for (int j = 0; j < N; ++j) {
    if (j == i) continue;
    float dx = xi - pos_x[j];
    float dy = yi - pos_y[j];
    if (fabsf(dx) < vr && fabsf(dy) < vr) {
      float sd = dx*dx + dy*dy;
      if (sd < pr2) {
        close_dx += dx;
        close_dy += dy;
      } else if (sd < vr2) {
        xavg += pos_x[j];
        yavg += pos_y[j];
        vxavg += vel_x[j];
        vyavg += vel_y[j];
        neighbors += 1;
      }
    }
  }

  if (neighbors > 0) {
    float inv = 1.0f / (float)neighbors;
    xavg *= inv; yavg *= inv; vxavg *= inv; vyavg *= inv;
    new_vx += (xavg - xi) * centering_factor + (vxavg - vel_x[i]) * matching_factor;
    new_vy += (yavg - yi) * centering_factor + (vyavg - vel_y[i]) * matching_factor;
  }

  new_vx += close_dx * avoid_factor;
  new_vy += close_dy * avoid_factor;

  // edges
  if (xi < margin) new_vx += turn_factor;
  if (xi > (float)W - margin) new_vx -= turn_factor;
  if (yi < margin) new_vy += turn_factor;
  if (yi > (float)H - margin) new_vy -= turn_factor;

  int bg = bias_group[i];
  float bv = bias_val[i];
  if (bg == 1) {
    new_vx = (1.0f - bv) * new_vx + bv * 1.0f;
  } else if (bg == -1) {
    new_vx = (1.0f - bv) * new_vx + bv * -1.0f;
  }

  // limit speed
  float sp = sqrtf(new_vx*new_vx + new_vy*new_vy);
  if (sp == 0.0f) sp = 1e-8f;
  if (sp > max_speed) {
    float s = max_speed / sp;
    new_vx *= s; new_vy *= s;
  }
  if (sp < min_speed) {
    float s = min_speed / sp;
    new_vx *= s; new_vy *= s;
  }

  // write back
  vel_x[i] = new_vx; vel_y[i] = new_vy;
  pos_x[i] = xi + new_vx * dt;
  pos_y[i] = yi + new_vy * dt;
}
}
"""


def main():
  # create initial state
  pos, vel, bias_group, bias_val = create_boids(N)

  # split into component arrays (float32)
  pos_x = pos[:,0].astype(np.float32).copy()
  pos_y = pos[:,1].astype(np.float32).copy()
  vel_x = vel[:,0].astype(np.float32).copy()
  vel_y = vel[:,1].astype(np.float32).copy()

  # device arrays
  pos_x_dev = gpuarray.to_gpu(pos_x)
  pos_y_dev = gpuarray.to_gpu(pos_y)
  vel_x_dev = gpuarray.to_gpu(vel_x)
  vel_y_dev = gpuarray.to_gpu(vel_y)
  bias_group_dev = gpuarray.to_gpu(bias_group.astype(np.int32))
  bias_val_dev = gpuarray.to_gpu(bias_val.astype(np.float32))

  # compile kernel
  mod = SourceModule(kernel_code)
  update = mod.get_function("update_boids")

  # debug prints: environment and initial state
  try:
    dev = cuda.Device(0)
    print(f"CUDA device: {dev.name()}")
  except Exception as e:
    print("Could not query CUDA device:", e)
  print(f"N={N}, W={W}, H={H}")
  print("pos x range:", pos_x.min(), pos_x.max())
  print("pos y range:", pos_y.min(), pos_y.max())
  print("vel x range:", vel_x.min(), vel_x.max())
  print("vel y range:", vel_y.min(), vel_y.max())
  print("biased counts:", np.sum(bias_group==1), np.sum(bias_group==-1))

  # plotting setup - use dark background and brighter boids so they're visible
  fig, ax = plt.subplots(figsize=(10, 7))
  fig.patch.set_facecolor('#07111a')
  ax.set_facecolor('#07111a')
  scat = ax.scatter(pos_x, pos_y, s=6.0, c='#4aa3ff')
  ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect('equal')
  ax.set_xticks([]); ax.set_yticks([])
  # FPS text: high-contrast boxed text so it's readable on dark background
  fps_text = ax.text(0.02, 0.97, "", transform=ax.transAxes, color='#e6eef6', fontsize=9,
                     ha='left', va='top', bbox=dict(facecolor='#0f1720', edgecolor='none', alpha=0.85))
  # ensure title is visible
  ax.set_title(f"GPU Boids N={N}", color='#e6eef6')

  # kernel launch params
  threads_per_block = 256
  blocks = (N + threads_per_block - 1) // threads_per_block

  frame_times = []

  def animate(frame):
    t0 = time.perf_counter()
    # launch kernel: arguments must match signature
    update(
      pos_x_dev, pos_y_dev,
      vel_x_dev, vel_y_dev,
      bias_group_dev, bias_val_dev,
      np.int32(N),
      np.float32(visual_range), np.float32(protected_range),
      np.float32(centering_factor), np.float32(avoid_factor), np.float32(matching_factor),
      np.float32(turn_factor), np.float32(margin), np.float32(min_speed), np.float32(max_speed), np.float32(dt),
      np.int32(W), np.int32(H),
      block=(threads_per_block, 1, 1), grid=(blocks, 1)
    )

    # copy back positions for plotting
    px = pos_x_dev.get()
    py = pos_y_dev.get()
    coords = np.column_stack((px, py))
    scat.set_offsets(coords)

    # debugging: print a few values on the first frames
    if frame < 3:
      print(f"frame={frame}: px[0:5]={px[:5]}, py[0:5]={py[:5]}")
      print(f"frame={frame}: px range: {px.min()}..{px.max()}, py range: {py.min()}..{py.max()}")
      if np.isnan(px).any() or np.isnan(py).any():
        print("NaNs detected in positions!")

    t1 = time.perf_counter()
    frame_times.append((t1 - t0) * 1000.0)
    if len(frame_times) > 200:
      frame_times.pop(0)
    # update title with FPS
    if frame_times:
      recent = np.array(frame_times[-30:])
      avg_ms = recent.mean() if recent.size else 0.0
      fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
      # update boxed FPS text for readability on dark background
      fps_text.set_text(f"FPS: {fps:.1f}")
      print(fps)
    return scat,

  ani = animation.FuncAnimation(fig, animate, interval=1, blit=False)
  plt.show()


if __name__ == '__main__':
  main()

