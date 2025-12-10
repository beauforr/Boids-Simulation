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
  // --- Thread indexing & bounds check ---
  // Each thread updates one boid; compute index and exit if out of range.
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  // --- Load boid state (position & velocity) ---
  // Read current position and velocity into local variables for computation.
  float xi = pos_x[i];
  float yi = pos_y[i];
  float new_vx = vel_x[i];
  float new_vy = vel_y[i];

  // --- Initialize accumulators for neighbor-based rules ---
  // Variables used to accumulate data for separation, cohesion and alignment.
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

  // --- Perception loop: find neighbors & accumulate contributions ---
  // Iterate over all boids, compute distances and accumulate:
  //   - separation: sum of displacement away for too-close boids
  //   - cohesion: average position of neighbors within visual range
  //   - alignment: average velocity of neighbors within visual range
  for (int j = 0; j < N; ++j) {
    if (j == i) continue;
    float dx = xi - pos_x[j];
    float dy = yi - pos_y[j];
    if (fabsf(dx) < vr && fabsf(dy) < vr) {
      float sd = dx*dx + dy*dy;
      if (sd < pr2) {
        // Too close -> separation accumulator
        close_dx += dx;
        close_dy += dy;
      } else if (sd < vr2) {
        // Within vision -> accumulate for cohesion & alignment
        xavg += pos_x[j];
        yavg += pos_y[j];
        vxavg += vel_x[j];
        vyavg += vel_y[j];
        neighbors += 1;
      }
    }
  }

  // --- Apply cohesion and alignment contributions (if any neighbors) ---
  // Move toward average position (cohesion) and match average velocity (alignment).
  if (neighbors > 0) {
    float inv = 1.0f / (float)neighbors;
    xavg *= inv; yavg *= inv; vxavg *= inv; vyavg *= inv;
    new_vx += (xavg - xi) * centering_factor + (vxavg - vel_x[i]) * matching_factor;
    new_vy += (yavg - yi) * centering_factor + (vyavg - vel_y[i]) * matching_factor;
  }

  // --- Apply separation contribution ---
  // Push away from too-close boids.
  new_vx += close_dx * avoid_factor;
  new_vy += close_dy * avoid_factor;

  // --- Edge avoidance / turning behavior ---
  // If boid nears the simulation boundary, apply a turn force to steer back.
  if (xi < margin) new_vx += turn_factor;
  if (xi > (float)W - margin) new_vx -= turn_factor;
  if (yi < margin) new_vy += turn_factor;
  if (yi > (float)H - margin) new_vy -= turn_factor;

  // --- Apply group bias (if any) ---
  // Some boids can have a bias to push their x-velocity toward +/-1.
  int bg = bias_group[i];
  float bv = bias_val[i];
  if (bg == 1) {
    new_vx = (1.0f - bv) * new_vx + bv * 1.0f;
  } else if (bg == -1) {
    new_vx = (1.0f - bv) * new_vx + bv * -1.0f;
  }

  // --- Speed limiting (min/max) ---
  // Enforce minimum and maximum speeds by scaling the velocity vector.
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

  // --- Write updated velocity and position back to global memory ---
  vel_x[i] = new_vx; vel_y[i] = new_vy;
  pos_x[i] = xi + new_vx * dt;
  pos_y[i] = yi + new_vy * dt;
}


