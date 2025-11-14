# boids_cpu.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, RadioButtons
import time
import matplotlib.gridspec as gridspec

# dark mode palette / rc settings
BG = "#0b0f1a"        # figure background
AX_BG = "#07111a"     # main axes background
PANEL_BG = "#0f1720"  # control panel / widget axes
WIDGET_BG = "#111827"
ACCENT = "#4aa3ff"    # accent color for sliders / active radio
TEXT = "#e6eef6"      # general text color

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": AX_BG,
    "axes.edgecolor": TEXT,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "text.color": TEXT,
    "savefig.facecolor": BG,
    "grid.color": "#222832"
})

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

# marker size (points^2 used by scatter 's' argument)
marker_size = 2.0

# mouse-follow parameters
mouse_active = False
mouse_pos = np.array([W/2.0, H/2.0])
mouse_strength = 0.006   # steering magnitude scale (tune)
mouse_range = 500.0      # pixels / units of the simulation

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

# ---------- Boids state (replaced with re-usable initializer) ----------
def create_boids(num):
    global N, pos, angles, vel, bias_group, bias_val
    N = int(max(1, round(num)))
    # positions and velocities as Nx arrays
    pos = np.random.rand(N, 2) * np.array([W, H])
    angles = np.random.rand(N) * 2 * np.pi
    vel = np.column_stack((np.cos(angles), np.sin(angles)))
    # scale initial velocities to avg speed ~ (min+max)/2
    v0 = (min_speed + max_speed) / 2.0
    vel *= v0
    bias_group = np.zeros(N, dtype=np.int8)  # 0 = none, 1 = right, -1 = left
    bias_val = np.full(N, default_bias)
    # do not randomize bias here; apply_bias_randomization() will be called later when UI is ready

# create initial boids
create_boids(N)

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

# use GridSpec to create a left main plot and a right controls column
# Slider configuration: (label, min, max, init)
slider_specs = [
    ("N", 10, 5000, N),
    ("marker_size", 1.0, 40.0, marker_size),
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

n_sliders = len(slider_specs)
# rows: 1 for radio, n_sliders for sliders, 1 for button, plus a small buffer row
n_rows = max(6, 1 + n_sliders + 1 + 1)
fig = plt.figure(figsize=(fig_w, fig_h))
gs = gridspec.GridSpec(n_rows, 2, width_ratios=[4, 1], figure=fig, wspace=0.15, hspace=0.6)

# main axes on the left spanning all rows
ax = fig.add_subplot(gs[:, 0])
ax.set_facecolor(AX_BG)
# use an explicit sizes array so we can update it at runtime
sizes = np.full(N, marker_size)
scat = ax.scatter(pos[:,0], pos[:,1], s=sizes, linewidths=0, alpha=0.95, c='#ffffff')
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal', adjustable='box')
ax.set_title("Boids CPU (method={}) N={}".format(NEIGHBOR_METHOD, N), fontsize=10, color=TEXT)

# add FPS display text (top-left)
fps_text = ax.text(0.02, 0.97, "", transform=ax.transAxes, color=TEXT, fontsize=9,
                   ha='left', va='top', bbox=dict(facecolor=AX_BG, edgecolor='none', alpha=0.6))

# controls column on the right: radio at top (row 0), sliders rows 1..n_sliders, button at last non-buffer row
ax_radio = fig.add_subplot(gs[0, 1])
ax_radio.set_facecolor(PANEL_BG)
radio = RadioButtons(ax_radio, ('grid', 'naive'), active=0 if NEIGHBOR_METHOD=='grid' else 1, activecolor=ACCENT)
# ensure radio labels are visible on dark bg
for lbl in radio.labels:
    lbl.set_color(TEXT)
# style slider axes and widgets
sliders = {}
for i, (name, mn, mx, init) in enumerate(slider_specs):
    ax_sl = fig.add_subplot(gs[1 + i, 1])
    ax_sl.set_facecolor(PANEL_BG)
    s = Slider(ax_sl, name, mn, mx, valinit=float(init), valfmt='%1.4f')
    # widget text colors
    try:
        s.label.set_color(TEXT)
        s.valtext.set_color(TEXT)
        s.ax.patch.set_facecolor(PANEL_BG)
        s.poly.set_facecolor(ACCENT)
    except Exception:
        pass
    sliders[name] = s

# place button near the bottom of the controls column
ax_button = fig.add_subplot(gs[1 + n_sliders, 1])
ax_button.set_facecolor(PANEL_BG)
btn = Button(ax_button, 'Randomize Bias', color="#1f2a35", hovercolor="#26333f")
# button label color
try:
    btn.label.set_color(TEXT)
except Exception:
    pass

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
    global fraction_biased, max_bias, N, marker_size, sizes
    if name == "N":
        newN = int(max(1, round(float(val))))
        if newN != N:
            create_boids(newN)
            # apply bias after re-creating arrays
            apply_bias_randomization()
            # update scatter offsets and sizes for new N
            try:
                scat.set_offsets(pos)
                sizes = np.full(N, marker_size)
                scat.set_sizes(sizes)
            except NameError:
                pass
            # update title to reflect new N
            try:
                ax.set_title("Boids CPU (method={}) N={}".format(NEIGHBOR_METHOD, N), color=TEXT)
            except NameError:
                pass
        return
    if name == "marker_size":
        marker_size = float(val)
        try:
            sizes = np.full(N, marker_size)
            scat.set_sizes(sizes)
        except NameError:
            pass
        return
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
    ax.set_title("Boids CPU (method={}) N={}".format(NEIGHBOR_METHOD, N), color=TEXT)
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
    # compute short-term average (last 30 frames) for stable FPS reading
    if frame_times:
        recent = frame_times[-30:]
        avg_ms = np.mean(recent)
        fps = (1000.0 / avg_ms) if avg_ms > 0 else 0.0
        fps_text.set_text(f"FPS: {fps:.1f}")
    if frame_count % 30 == 0:
        avg = np.mean(frame_times[-30:])
        print(f"avg frame time (last 30): {avg:.2f} ms")
    scat.set_offsets(pos)
    return scat, fps_text

ani = animation.FuncAnimation(fig, animate, frames=10000, interval=1, blit=True)
plt.show()

# Print summary on close
end_time = time.perf_counter()
duration = end_time - start_time
if frame_times:
    print("Overall avg frame time (ms):", np.mean(frame_times))
    print("Median frame time (ms):", np.median(frame_times))
    print("Total frames:", frame_count, "wall-clock sec:", duration)
    print("FPS (frames/sec):", frame_count / duration)
