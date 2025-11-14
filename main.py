# boids_cpu.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, RadioButtons
import time


# JE MOET DINGEN UIT HET LABO GEBRUIKEN !!!!!!!!

# BENCHMARKS, VERSCHILLEN THREADS ETC ETC

# ---------- Parameters ----------
W, H = 2000, 1500
N = 1500                      # number of boids (tune)
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

# bias groups (some boids have bias to left (-1) or right (+1))
fraction_biased = 0.05
max_bias = 0.01
default_bias = 0.001

# choose neighbor method: 'naive' or 'grid'
NEIGHBOR_METHOD = 'grid'

# ---------- Utilities ----------
def limit_speed(vx, vy, minspeed, maxspeed):
    sp = np.hypot(vx, vy)
    # prevent zero
    sp = np.where(sp == 0, 1e-8, sp)
    too_fast = sp > maxspeed
    too_slow = sp < minspeed
    scale = np.ones_like(sp)
    scale = np.where(too_fast, maxspeed / sp, scale)
    scale = np.where(too_slow, minspeed / sp, scale)
    return vx * scale, vy * scale

# ---------- Boids state ----------
# positions and velocities as Nx arrays
pos = np.random.rand(N, 2) * np.array([W, H])
angles = np.random.rand(N) * 2 * np.pi
vel = np.column_stack((np.cos(angles), np.sin(angles)))
# scale initial velocities to avg speed ~ (min+max)/2
v0 = (min_speed + max_speed) / 2.0
vel *= v0

bias_group = np.zeros(N, dtype=np.int8)  # 0 = none, 1 = right, -1 = left
bias_val = np.full(N, default_bias)
# randomly choose some boids to be biased
num_biased = int(N * fraction_biased)
inds = np.random.choice(N, num_biased, replace=False)
# split half right, half left
half = num_biased // 2
bias_group[inds[:half]] = 1
bias_group[inds[half:]] = -1

# ---------- Spatial grid helper ----------
def make_grid_index(pos, cell_size):
    # returns integer grid coords (gx, gy) and 1D flat index
    g = np.floor(pos / cell_size).astype(int)
    return g

def grid_neighbors(pos, cell_size, vr):
    # Build a hash map of grid cell -> list of boid indices
    N = pos.shape[0]
    g = make_grid_index(pos, cell_size)
    keys = (g[:,0].astype(np.int64) << 32) ^ (g[:,1].astype(np.int64) & 0xffffffff)
    # naive python dict building (still much faster than O(N^2) for large N)
    cell_map = {}
    for i, key in enumerate(keys):
        cell_map.setdefault(key, []).append(i)
    # For each boid, look in neighbour cells and collect candidate indices
    neighbor_list = [[] for _ in range(N)]
    offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]
    for i in range(N):
        gx, gy = g[i]
        cand = []
        for ox, oy in offsets:
            key = ((gx+ox) << 32) ^ ((gy+oy) & 0xffffffff)
            if key in cell_map:
                cand.extend(cell_map[key])
        neighbor_list[i] = cand
    return neighbor_list

# ---------- Update functions ----------
def update_naive(pos, vel):
    N = pos.shape[0]
    new_vel = vel.copy()
    for i in range(N):
        close_dx = 0.0
        close_dy = 0.0
        xavg = 0.0
        yavg = 0.0
        vxavg = 0.0
        vyavg = 0.0
        neighbors = 0
        xi, yi = pos[i]
        for j in range(N):
            if i == j: continue
            dx = xi - pos[j,0]
            dy = yi - pos[j,1]
            if abs(dx) < visual_range and abs(dy) < visual_range:
                sd = dx*dx + dy*dy
                if sd < protected_range*protected_range:
                    close_dx += dx
                    close_dy += dy
                elif sd < visual_range*visual_range:
                    xavg += pos[j,0]
                    yavg += pos[j,1]
                    vxavg += vel[j,0]
                    vyavg += vel[j,1]
                    neighbors += 1
        if neighbors > 0:
            xavg /= neighbors
            yavg /= neighbors
            vxavg /= neighbors
            vyavg /= neighbors
            new_vel[i,0] += (xavg - xi) * centering_factor + (vxavg - vel[i,0]) * matching_factor
            new_vel[i,1] += (yavg - yi) * centering_factor + (vyavg - vel[i,1]) * matching_factor
        new_vel[i,0] += close_dx * avoid_factor
        new_vel[i,1] += close_dy * avoid_factor

        # edges
        if pos[i,0] < margin: new_vel[i,0] += turn_factor
        if pos[i,0] > W - margin: new_vel[i,0] -= turn_factor
        if pos[i,1] < margin: new_vel[i,1] += turn_factor
        if pos[i,1] > H - margin: new_vel[i,1] -= turn_factor

        # bias
        if bias_group[i] == 1:
            new_vel[i,0] = (1 - bias_val[i]) * new_vel[i,0] + bias_val[i] * 1.0
        elif bias_group[i] == -1:
            new_vel[i,0] = (1 - bias_val[i]) * new_vel[i,0] + bias_val[i] * -1.0

    # limit speed
    new_vx, new_vy = limit_speed(new_vel[:,0], new_vel[:,1], min_speed, max_speed)
    new_vel[:,0] = new_vx
    new_vel[:,1] = new_vy
    pos[:] = pos + new_vel * dt
    vel[:] = new_vel

def update_grid(pos, vel):
    cell_size = visual_range
    neighbor_list = grid_neighbors(pos, cell_size, visual_range)
    N = pos.shape[0]
    new_vel = vel.copy()
    for i in range(N):
        xi, yi = pos[i]
        close_dx = 0.0
        close_dy = 0.0
        xavg = yavg = vxavg = vyavg = 0.0
        neighbors = 0
        for j in neighbor_list[i]:
            if j == i: continue
            dx = xi - pos[j,0]
            dy = yi - pos[j,1]
            # exact circular check
            sd = dx*dx + dy*dy
            if sd < protected_range*protected_range:
                close_dx += dx
                close_dy += dy
            elif sd < visual_range*visual_range:
                xavg += pos[j,0]
                yavg += pos[j,1]
                vxavg += vel[j,0]
                vyavg += vel[j,1]
                neighbors += 1
        if neighbors > 0:
            xavg /= neighbors
            yavg /= neighbors
            vxavg /= neighbors
            vyavg /= neighbors
            new_vel[i,0] += (xavg - xi) * centering_factor + (vxavg - vel[i,0]) * matching_factor
            new_vel[i,1] += (yavg - yi) * centering_factor + (vyavg - vel[i,1]) * matching_factor
        new_vel[i,0] += close_dx * avoid_factor
        new_vel[i,1] += close_dy * avoid_factor

        # edges
        if pos[i,0] < margin: new_vel[i,0] += turn_factor
        if pos[i,0] > W - margin: new_vel[i,0] -= turn_factor
        if pos[i,1] < margin: new_vel[i,1] += turn_factor
        if pos[i,1] > H - margin: new_vel[i,1] -= turn_factor

        # bias
        if bias_group[i] == 1:
            new_vel[i,0] = (1 - bias_val[i]) * new_vel[i,0] + bias_val[i] * 1.0
        elif bias_group[i] == -1:
            new_vel[i,0] = (1 - bias_val[i]) * new_vel[i,0] + bias_val[i] * -1.0

    # limit speed
    new_vx, new_vy = limit_speed(new_vel[:,0], new_vel[:,1], min_speed, max_speed)
    new_vel[:,0] = new_vx
    new_vel[:,1] = new_vy
    pos[:] = pos + new_vel * dt
    vel[:] = new_vel

# ---------- Animation / main loop with interactive controls ----------
# use a reasonable figure size and smaller markers to reduce overlap
fig_w = min(12, W / 150.0)
fig_h = min(9, H / 150.0)
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
# smaller marker size, no edge linewidth, slight alpha to reduce dense-overlap look
scat = ax.scatter(pos[:,0], pos[:,1], s=2, linewidths=0, alpha=0.85)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal', adjustable='box')
ax.set_title("Boids CPU (method={}) N={}".format(NEIGHBOR_METHOD, N), fontsize=10)

# reserve right column for controls and room at bottom for sliders
fig.subplots_adjust(left=0.05, right=0.75, top=0.95, bottom=0.38)

# Slider configuration: (label, min, max, init)
slider_specs = [
    ("visual_range", 5.0, 200.0, visual_range),
    ("protected_range", 1.0, 50.0, protected_range),
    ("centering_factor", 0.0, 0.01, centering_factor),
    ("avoid_factor", 0.0, 0.2, avoid_factor),
    ("matching_factor", 0.0, 0.2, matching_factor),
    ("turn_factor", 0.0, 1.0, turn_factor),
    ("min_speed", 0.0, 10.0, min_speed),
    ("max_speed", 0.1, 20.0, max_speed),
    ("fraction_biased", 0.0, 0.5, fraction_biased),
    ("max_bias", 0.0, 0.05, max_bias),
]

# place controls in the reserved right column and center them vertically
sliders = {}
n_sliders = len(slider_specs)
height = 0.025
gap = 0.01
control_x = 0.76
control_w = 0.22
# define the vertical control area (within full figure coords)
control_top = 0.92
control_bottom = 0.06
control_height = control_top - control_bottom

# sizes for radio and button
radio_h = 0.06
btn_h = 0.06
spacing = 0.02  # spacing between groups

# total height needed: radio + spacing + sliders stack + spacing + button
sliders_stack_h = n_sliders * height + (n_sliders - 1) * gap if n_sliders > 0 else 0.0
total_needed = radio_h + spacing + sliders_stack_h + spacing + btn_h

# compute start so the whole block is vertically centered in control area
start_y = control_bottom + max(0.0, (control_height - total_needed) / 2.0)

# place radio at the top of the block
radio_y = start_y + sliders_stack_h + spacing + btn_h + spacing if False else start_y + sliders_stack_h + spacing + btn_h + spacing
# (simplify: place radio above sliders)
radio_y = start_y + sliders_stack_h + spacing + btn_h + spacing - btn_h - spacing + radio_h - radio_h
# correct simpler placement:
radio_y = start_y + sliders_stack_h + spacing + btn_h + spacing - radio_h
ax_radio = fig.add_axes([control_x, radio_y, control_w, radio_h], facecolor='lightgoldenrodyellow')
radio = RadioButtons(ax_radio, ('grid', 'naive'), active=0 if NEIGHBOR_METHOD=='grid' else 1)

# place sliders stacked starting at start_y (bottom-up)
for i, (name, mn, mx, init) in enumerate(slider_specs):
    axpos = [control_x, start_y + i * (height + gap), control_w, height]
    ax_sl = fig.add_axes(axpos)
    s = Slider(ax_sl, name, mn, mx, valinit=float(init), valfmt='%1.4f')
    sliders[name] = s

# place button below the sliders block (aligned with control_x)
btn_y = start_y + sliders_stack_h + spacing
ax_button = fig.add_axes([control_x, btn_y, control_w, btn_h])
btn = Button(ax_button, 'Randomize Bias', color='lightblue', hovercolor='0.975')

# helper to (re)assign bias groups/values
def apply_bias_randomization():
    global bias_group, bias_val, fraction_biased, max_bias
    num_biased = int(N * fraction_biased)
    bias_group[:] = 0
    if num_biased > 0:
        inds = np.random.choice(N, num_biased, replace=False)
        half = num_biased // 2
        bias_group[inds[:half]] = 1
        bias_group[inds[half:]] = -1
        # bias strength per biased boid uniformly in [default_bias, max_bias]
        if max_bias <= default_bias:
            bias_val[:] = default_bias
        else:
            vals = np.random.rand(num_biased) * (max_bias - default_bias) + default_bias
            bias_val[:] = default_bias
            bias_val[inds] = vals

# initial bias application (respect current fraction/max_bias)
apply_bias_randomization()

# callbacks for sliders
def on_slider_change(val, name):
    global visual_range, protected_range, centering_factor, avoid_factor
    global matching_factor, turn_factor, min_speed, max_speed
    global fraction_biased, max_bias
    if name == "visual_range":
        visual_range = float(val)
    elif name == "protected_range":
        protected_range = float(val)
    elif name == "centering_factor":
        centering_factor = float(val)
    elif name == "avoid_factor":
        avoid_factor = float(val)
    elif name == "matching_factor":
        matching_factor = float(val)
    elif name == "turn_factor":
        turn_factor = float(val)
    elif name == "min_speed":
        min_speed = float(val)
        # ensure min <= max
        if min_speed >= max_speed:
            max_speed = min_speed + 0.1
            sliders["max_speed"].set_val(max_speed)
    elif name == "max_speed":
        max_speed = float(val)
        if max_speed <= min_speed:
            min_speed = max_speed - 0.1
            if min_speed < 0.0:
                min_speed = 0.0
            sliders["min_speed"].set_val(min_speed)
    elif name == "fraction_biased":
        fraction_biased = float(val)
        apply_bias_randomization()
    elif name == "max_bias":
        max_bias = float(val)
        apply_bias_randomization()

# attach callbacks
for name, s in sliders.items():
    s.on_changed(lambda v, n=name: on_slider_change(v, n))

def radio_changed(label):
    global NEIGHBOR_METHOD
    NEIGHBOR_METHOD = label
    ax.set_title("Boids CPU (method={}) N={}".format(NEIGHBOR_METHOD, N))
radio.on_clicked(radio_changed)

def button_clicked(event):
    apply_bias_randomization()
btn.on_clicked(button_clicked)

# benchmarking
frame_times = []
frame_count = 0
start_time = time.perf_counter()

def animate(frame):
    global frame_count
    t0 = time.perf_counter()
    if NEIGHBOR_METHOD == 'naive':
        update_naive(pos, vel)
    else:
        update_grid(pos, vel)
    t1 = time.perf_counter()
    frame_times.append((t1 - t0)*1000.0)
    frame_count += 1
    if frame_count % 30 == 0:
        avg = np.mean(frame_times[-30:])
        print(f"avg frame time (last 30): {avg:.2f} ms")
    scat.set_offsets(pos)
    return scat,

ani = animation.FuncAnimation(fig, animate, frames=10000, interval=1, blit=True)
plt.show()

# Print summary on close
end_time = time.perf_counter()
duration = end_time - start_time
if frame_times:
    print("Overall avg frame time (ms):", np.mean(frame_times))
    print("Median frame time (ms):", np.median(frame_times))
    print("Total frames:", frame_count, "wall-clock sec:", duration)
