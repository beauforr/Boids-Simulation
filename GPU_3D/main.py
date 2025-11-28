import math
import numpy as np
import time
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# ---------- Simulation parameters ----------
W, H, D = 1500, 6000, 1500
N = 3000

# Prey Physics
visual_range = 250.0  
protected_range = 40.0
centering_factor = 0.0005
avoid_factor = 0.05
hostile_factor = 0.15
matching_factor = 0.05
turn_factor = 0.2
margin = 100.0
min_speed = 4.0
max_speed = 9.0
dt = 1.0
fov_angle = -0.2

# --- PREDATOR PARAMETERS ---
predator_speed = 11.0 # Slightly faster than prey max speed
# How far away prey spot the predator (should be large)
predator_fear_range = 500.0 
# How strongly they run away (must override local flocking)
predator_avoid_factor = 2.5 

# ---------- Helpers / initialization ----------
def create_boids_3d(num):
  num = int(max(1, round(num)))
  pos = np.random.rand(num, 3).astype(np.float32) * np.array([W, H, D], dtype=np.float32)
  
  phi = np.random.rand(num).astype(np.float32) * 2.0 * np.pi
  costheta = 2.0 * np.random.rand(num).astype(np.float32) - 1.0
  theta = np.arccos(costheta)
  vel = np.column_stack((
      np.sin(theta) * np.cos(phi), 
      np.sin(theta) * np.sin(phi), 
      costheta
  )).astype(np.float32)
  vel *= (min_speed + max_speed) / 2.0
  
  bias_group = np.random.randint(0, 4, size=num).astype(np.int32)
  bias_val = np.zeros(num, dtype=np.float32)
  return pos, vel, bias_group, bias_val

# Initialize Predator (CPU side is fine for just one)
pred_pos = np.array([W/2, H/2, D/2], dtype=np.float32)
# Start moving in a random direction
theta_p = np.random.rand() * np.pi
phi_p = np.random.rand() * 2 * np.pi
pred_vel = np.array([
    np.sin(theta_p)*np.cos(phi_p),
    np.sin(theta_p)*np.sin(phi_p),
    np.cos(theta_p)
], dtype=np.float32) * predator_speed


# ---------- CUDA kernel (With Predator Fear Logic) ----------
kernel_code = r"""
#include <math.h>
extern "C" {
__global__ void update_boids_3d(
  float *pos_x, float *pos_y, float *pos_z,
  float *vel_x, float *vel_y, float *vel_z,
  int *bias_group, float *bias_val,
  int N,
  float visual_range, float protected_range,
  float centering_factor, float avoid_factor, float matching_factor,
  float turn_factor, float margin, float min_speed, float max_speed, float dt,
  int W, int H, int D, float fov_angle, float hostile_factor,
  float pred_x, float pred_y, float pred_z, float pred_range, float pred_avoid_factor)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  float xi = pos_x[i];
  float yi = pos_y[i];
  float zi = pos_z[i];
  float vxi = vel_x[i];
  float vyi = vel_y[i];
  float vzi = vel_z[i];
  int my_group = bias_group[i];

  float speed_sq = vxi*vxi + vyi*vyi + vzi*vzi;
  float speed = sqrtf(speed_sq);
  if (speed == 0.0f) speed = 0.0001f;

  float close_dx = 0.0f; float close_dy = 0.0f; float close_dz = 0.0f;
  float xavg = 0.0f; float yavg = 0.0f; float zavg = 0.0f;
  float vxavg = 0.0f; float vyavg = 0.0f; float vzavg = 0.0f;
  int neighbors = 0;

  float pr2 = protected_range * protected_range;
  float vr2 = visual_range * visual_range;

  // --- 1. Standard Boid Interactions ---
  for (int j = 0; j < N; ++j) {
    if (j == i) continue;
    float dx = xi - pos_x[j];
    float dy = yi - pos_y[j];
    float dz = zi - pos_z[j];
    float dist2 = dx*dx + dy*dy + dz*dz;
    int other_group = bias_group[j];
    bool is_friend = (my_group == other_group);

    if (dist2 < pr2) {
      if (is_friend) { close_dx += dx; close_dy += dy; close_dz += dz; } 
      else {
        float multiplier = hostile_factor / avoid_factor; 
        close_dx += dx * multiplier; close_dy += dy * multiplier; close_dz += dz * multiplier;
      }
    } else if (dist2 < vr2 && is_friend) {
      float dot = ((-dx * vxi) + (-dy * vyi) + (-dz * vzi));
      float cosine = dot / (speed * sqrtf(dist2));
      if (cosine > fov_angle) {
          xavg += pos_x[j]; yavg += pos_y[j]; zavg += pos_z[j];
          vxavg += vel_x[j]; vyavg += vel_y[j]; vzavg += vel_z[j];
          neighbors += 1;
      }
    }
  }

  float new_vx = vxi; float new_vy = vyi; float new_vz = vzi;

  if (neighbors > 0) {
    float inv = 1.0f / (float)neighbors;
    new_vx += ((xavg * inv) - xi) * centering_factor + ((vxavg * inv) - vxi) * matching_factor;
    new_vy += ((yavg * inv) - yi) * centering_factor + ((vyavg * inv) - vyi) * matching_factor;
    new_vz += ((zavg * inv) - zi) * centering_factor + ((vzavg * inv) - vzi) * matching_factor;
  }
  new_vx += close_dx * avoid_factor;
  new_vy += close_dy * avoid_factor;
  new_vz += close_dz * avoid_factor;

  // --- 2. PREDATOR AVOIDANCE ---
  // Vector FROM predator TO boid
  float pdx = xi - pred_x;
  float pdy = yi - pred_y;
  float pdz = zi - pred_z;
  float pdist2 = pdx*pdx + pdy*pdy + pdz*pdz;

  if (pdist2 < pred_range * pred_range) {
    // Normalize the flee vector to ensure consistent strong force
    float pdist = sqrtf(pdist2);
    // Avoid division by zero if exactly on top of predator
    if (pdist < 0.01f) pdist = 0.01f; 
    float inv_pdist = 1.0f / pdist;

    // Apply strong repulsion force away from predator
    new_vx += (pdx * inv_pdist) * pred_avoid_factor;
    new_vy += (pdy * inv_pdist) * pred_avoid_factor;
    new_vz += (pdz * inv_pdist) * pred_avoid_factor;
  }

  // Box Boundary
  if (xi < margin) new_vx += turn_factor;
  if (xi > (float)W - margin) new_vx -= turn_factor;
  if (yi < margin) new_vy += turn_factor;
  if (yi > (float)H - margin) new_vy -= turn_factor;
  if (zi < margin) new_vz += turn_factor;
  if (zi > (float)D - margin) new_vz -= turn_factor;

  // Speed Limit
  float sp = sqrtf(new_vx*new_vx + new_vy*new_vy + new_vz*new_vz);
  if (sp == 0.0f) sp = 1e-8f;
  if (sp > max_speed) {
    float s = max_speed / sp; new_vx *= s; new_vy *= s; new_vz *= s;
  } else if (sp < min_speed) {
    float s = min_speed / sp; new_vx *= s; new_vy *= s; new_vz *= s;
  }

  vel_x[i] = new_vx; vel_y[i] = new_vy; vel_z[i] = new_vz;
  pos_x[i] = xi + new_vx * dt;
  pos_y[i] = yi + new_vy * dt;
  pos_z[i] = zi + new_vz * dt;
}
}
"""

def draw_simulation_box(ax, width, height, depth):
    color = "#000000"
    alpha = 0.2
    edges = [([0, W], [0, 0], [0, 0]), ([W, W], [0, H], [0, 0]), ([W, 0], [H, H], [0, 0]), ([0, 0], [H, 0], [0, 0]), ([0, W], [0, 0], [D, D]), ([W, W], [0, H], [D, D]), ([W, 0], [H, H], [D, D]), ([0, 0], [H, 0], [D, D]), ([0, 0], [0, 0], [0, D]), ([W, W], [0, 0], [0, D]), ([W, W], [H, H], [0, D]), ([0, 0], [H, H], [0, D])]
    for (xs, ys, zs) in edges: ax.plot(xs, ys, zs, c=color, alpha=alpha, linewidth=1.0)

def main():
  pos, vel, bias_group, bias_val = create_boids_3d(N)
  pos_x_dev = gpuarray.to_gpu(pos[:,0].astype(np.float32))
  pos_y_dev = gpuarray.to_gpu(pos[:,1].astype(np.float32))
  pos_z_dev = gpuarray.to_gpu(pos[:,2].astype(np.float32))
  vel_x_dev = gpuarray.to_gpu(vel[:,0].astype(np.float32))
  vel_y_dev = gpuarray.to_gpu(vel[:,1].astype(np.float32))
  vel_z_dev = gpuarray.to_gpu(vel[:,2].astype(np.float32))
  bias_group_dev = gpuarray.to_gpu(bias_group.astype(np.int32))
  bias_val_dev = gpuarray.to_gpu(bias_val.astype(np.float32))

  mod = SourceModule(kernel_code)
  update = mod.get_function("update_boids_3d")

  colors = np.zeros((N, 4))
  for i in range(N):
      bg = bias_group[i]
      if bg == 0:   colors[i] = [1.0, 0.9, 0.0, 1.0] # Electric Yellow (Yellow Tang)
      elif bg == 1: colors[i] = [1.0, 0.4, 0.2, 1.0] # Deep Coral (Clownfish)
      elif bg == 2: colors[i] = [0.2, 0.5, 1.0, 1.0] # Neon Blue (Blue Tang)
      else:         colors[i] = [0.0, 1.0, 0.5, 1.0] # Seafoam Green (Anemone/Algae)

  fig = plt.figure(figsize=(10, 8))
  fig.patch.set_facecolor("#acb3b9")
  ax = fig.add_subplot(111, projection='3d')
  ax.set_facecolor('#acb3b9')
  ax.xaxis.set_pane_color((0,0,0,0)); ax.yaxis.set_pane_color((0,0,0,0)); ax.zaxis.set_pane_color((0,0,0,0))
  draw_simulation_box(ax, W, H, D)
  ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_zlim(0, D)
  max_range = np.array([W, H, D]).max()
  ax.set_box_aspect((W/max_range, H/max_range, D/max_range))
  ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
  ax.grid(False)

  # Prey scatter
  scat = ax.scatter(pos[:,0], pos[:,1], pos[:,2], s=5.0, c=colors, depthshade=True)
  # PREDATOR scatter (Big Red Dot)
  scat_pred = ax.scatter([pred_pos[0]], [pred_pos[1]], [pred_pos[2]], s=150.0, c='red', edgecolors='white', depthshade=False)

  ax.set_title(f"3D Boids vs Predator", color='#e6eef6')
  fps_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, color='#e6eef6')

  threads_per_block = 128
  blocks = (N + threads_per_block - 1) // threads_per_block
  frame_times = []

  def animate(frame):
    t0 = time.perf_counter()
    
    # 1. Update Predator (CPU side simplest for single entity)
    global pred_pos, pred_vel
    pred_pos += pred_vel * dt
    # Bounce predator off walls
    if pred_pos[0] < 0 or pred_pos[0] > W: pred_vel[0] *= -1
    if pred_pos[1] < 0 or pred_pos[1] > H: pred_vel[1] *= -1
    if pred_pos[2] < 0 or pred_pos[2] > D: pred_vel[2] *= -1
    # Clamp predator positions to stay strictly in box
    pred_pos = np.clip(pred_pos, [1,1,1], [W-1, H-1, D-1])

    # 2. Update Prey (GPU) - Pass predator coords to kernel
    update(
      pos_x_dev, pos_y_dev, pos_z_dev,
      vel_x_dev, vel_y_dev, vel_z_dev,
      bias_group_dev, bias_val_dev,
      np.int32(N),
      np.float32(visual_range), np.float32(protected_range),
      np.float32(centering_factor), np.float32(avoid_factor), np.float32(matching_factor),
      np.float32(turn_factor), np.float32(margin), np.float32(min_speed), np.float32(max_speed), np.float32(dt),
      np.int32(W), np.int32(H), np.int32(D), np.float32(fov_angle), np.float32(hostile_factor),
      # NEW: Predator arguments
      np.float32(pred_pos[0]), np.float32(pred_pos[1]), np.float32(pred_pos[2]), 
      np.float32(predator_fear_range), np.float32(predator_avoid_factor),
      block=(threads_per_block, 1, 1), grid=(blocks, 1)
    )

    # 3. Update Visuals
    px = pos_x_dev.get(); py = pos_y_dev.get(); pz = pos_z_dev.get()
    scat._offsets3d = (px, py, pz)
    # Update predator scatter position (needs list of coordinates)
    scat_pred._offsets3d = ([pred_pos[0]], [pred_pos[1]], [pred_pos[2]])

    t1 = time.perf_counter()
    frame_times.append((t1 - t0) * 1000.0)
    if len(frame_times) > 50: frame_times.pop(0)
    if frame_times:
        avg = sum(frame_times)/len(frame_times)
        fps_text.set_text(f"FPS: {1000.0/avg:.1f}")
    return scat, scat_pred

  ani = animation.FuncAnimation(fig, animate, interval=1, blit=False)
  plt.show()

if __name__ == '__main__':
  main()