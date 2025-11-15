#!/usr/bin/env python3
"""
This script works best for pixelation that uses nearest neighbour pixelation as opposed to
kernel averaging pixelation which will give more of a blur. I would recommend trying it out on 
a yt-dlp downloaded copy of the Jeff Geerling video https://www.youtube.com/watch?v=acKYYwcxpGk

Purpose
-------
Recover a higher-resolution (“super-resolution”) view of content seen through a
**stationary pixelation grid** (mosaic/blur grid) while a **window moves
underneath** it.

The script assumes:
- The grid spacing (cell period) is fixed (e.g. ~25.2 px).
- The grid is static in image coordinates.
- The window/region you care about translates over time (no rotation/scale).

High-level idea
---------------
Each frame shows the same underlying scene sampled at *different sub-pixel
phases* because the window slides under the stationary grid. We:
1) **Track the window** by template matching on an *edge ring* around it
   (robust to interior changes).
2) **Rebuild a canonical window** coordinate system (top-left aligned to the
   first frame).
3) For each frame, compute every **grid cell centre inside the tracked window**,
   sample the colour there, and **project it into the canonical window** at the
   corresponding location.
4) **Accumulate** samples using either *nearest* or *bilinear* “splatting”.
5) Optionally **fill holes** using a box-filter growth (sum/count) until fully
   covered.
6) Save the before-fill and final reconstructions plus optional debug overlays.

What you do (interactive steps)
-------------------------------
1) **Pick the moving window**  
   A resizable OpenCV window opens on the first frame.  
   - Drag a rectangle around the window region you want.  
   - Press **ENTER** to accept, **R** to reset, **ESC** to cancel.

2) **Set grid size & phase**  
   A second window shows the frame with an overlaid red grid you control.  
   - **Click one grid *intersection*** (corner where two grid lines meet).  
   - Adjust the **cell size** (spacing) and **phase** so the red lines match the
     visible grid precisely.

   Key bindings:
   - Cell size (both axes): **W/S** = ±1 px, **. / ,** = ±0.1 px  
   - Phase X/Y (grid line offset):  
     • **←/→** = ±1 px X, **↑/↓** = ±1 px Y  
     • **J/L** = ±0.1 px X, **I/K** = ±0.1 px Y  
   - **ENTER** accept, **R** reset, **ESC** cancel.

3) **Let it run**  
   The script tracks the window across frames, samples centre pixels, projects
   them into the canonical window, accumulates, and (optionally) fills holes.

Outputs
-------
- `reconstruction_sr_before_fill.png` — SR result from raw accumulation.
- `reconstruction_sr.png` — SR result after optional hole-fill.
- (Optional) Debug folder `debug_gridtrack/` with:
  - `overlays/grid_XXXXXX.png` — per-frame overlay (tracked window in yellow,
    red grid lines, **green dots at cell *centres***).
  - `tracking_log.csv` — frame index, tracked box, match response, sample count.

Key parameters (edit in the "USER SETTINGS" section)
----------------------------------------------------
- **VIDEO_PATH**: input video.
- **START_AT_FRAME / MAX_FRAMES / FRAME_STRIDE**: which frames to use.
- **CELL_SIZE / CELL_SIZE_Y**: measured grid period (can be non-integer).
- **SR_FACTOR**: 1 = native window resolution; 2–4 = true SR (more detail with
  enough motion, slower & more memory).
- **SR_SPLAT_MODE**: `"nearest"` (crisper but aliasy) or `"bilinear"` (smoother,
  better sub-pixel integration).
- **TRACK_SEARCH_MARGIN / TRACK_EDGE_RING / TRACK_MIN_RESPONSE**: robustness and
  speed of the window tracker. Increase margin if the window jumps; increase
  ring to rely more on the border; raise `TRACK_MIN_RESPONSE` to reject bad
  matches (falls back to previous position).
- **FILL_MAX_ITERS**: 0 to disable fill; higher for more aggressive coverage.
- **SAVE_DEBUG / SAVE_OVERLAY_EVERY_FRAME / OVERLAY_PERIOD**: control debug
  images and CSV logging.

How it works (algorithm details)
--------------------------------
- **Window tracking**: builds an *edge-magnitude ring* template from your picked
  ROI. Each new frame is searched in a padded region using normalized template
  matching. The peak location gives the new top-left; weak peaks retain the
  previous location.
- **Centre sampling**: for the current tracked box `(x, y, w, h)`, we generate
  the stationary grid’s **centre coordinates** inside the box:
"""

import os, cv2, math, csv, numpy as np
from pathlib import Path

# ───────────────────────── USER SETTINGS ─────────────────────────
VIDEO_PATH          = "targetvideo.mp4"
START_AT_FRAME      = 50
MAX_FRAMES          = 200
FRAME_STRIDE        = 1

# Stationary grid (measured)
CELL_SIZE           = 25.2      # X period in pixels (float OK)
CELL_SIZE_Y         = 25.2      # Y period in pixels; set different if rectangular

# Super-resolution canvas
SR_FACTOR           = 1         # 1=window native; 2–4 true SR (needs many frames)
SR_SPLAT_MODE       = "bilinear"  # "nearest" or "bilinear"

# Tracking (translation only)
TRACK_SEARCH_MARGIN = 300       # search band (px) around last top-left
TRACK_EDGE_RING     = 100        # border ring thickness for the template (px)
TRACK_MIN_RESPONSE  = 0.05      # fallback to previous position if peak below this

# Hole fill
FILL_MAX_ITERS      = 1200      # 0 = disable

# Debug
SAVE_DEBUG                = False
DEBUG_DIR                 = "debug_gridtrack"
SAVE_OVERLAY_EVERY_FRAME  = True   # True: all frames; False: every N frames
OVERLAY_PERIOD            = 10     # used only if SAVE_OVERLAY_EVERY_FRAME=False
OVERLAY_DRAW_CENTRES      = True   # draw green dots at centres
# ──────────────────────────────────────────────────────────────────

# Arrow keys (common OpenCV keycodes)
KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN = 2424832, 2555904, 2490368, 2621440

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)
def clamp(v, lo, hi): return max(lo, min(hi, v))

def to_gray_f32(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return g.astype(np.float32)

def draw_window(box, img, color=(0,255,255), thick=2):
    x, y, w, h = box
    cv2.rectangle(img, (x,y), (x+w-1,y+h-1), color, thick, cv2.LINE_AA)

# ───────────── Window picker (drag) ─────────────
def window_picker(frame_bgr, title="Pick window (drag). ENTER accept • R reset • ESC cancel",
                  max_w=1600, max_h=1000):
    H, W = frame_bgr.shape[:2]
    rect_xyxy, drag = None, None

    def scale():
        try:
            _,_,ww,hh = cv2.getWindowImageRect(title)
            s = min(ww / max(1,W), hh / max(1,H))
            return s if np.isfinite(s) and s > 0 else min(1.0, max_w/W, max_h/H)
        except Exception:
            return min(1.0, max_w/W, max_h/H)

    def draw(s):
        disp = cv2.resize(frame_bgr, (max(1,int(W*s)), max(1,int(H*s))), interpolation=cv2.INTER_AREA)
        hud = "Drag to box the window. ENTER accept • R reset • ESC cancel"
        cv2.putText(disp, hud, (10, max(24,int(24*s))), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.4,0.6*s), (0,255,255), max(1,int(2*s)), cv2.LINE_AA)
        if rect_xyxy:
            x1,y1,x2,y2 = rect_xyxy
            p1 = (int(round(x1*s)), int(round(y1*s)))
            p2 = (int(round(x2*s)), int(round(y2*s)))
            cv2.rectangle(disp, p1, p2, (0,255,255), max(1,int(2*s)), cv2.LINE_AA)
        return disp

    def on_mouse(ev, x, y, flags, _):
        nonlocal drag, rect_xyxy
        s = scale()
        if ev == cv2.EVENT_LBUTTONDOWN:
            drag = (int(round(x/s)), int(round(y/s)))
            rect_xyxy = (drag[0], drag[1], drag[0], drag[1])
        elif ev == cv2.EVENT_MOUSEMOVE and drag is not None:
            x0,y0 = drag
            x1 = int(round(x/s)); y1 = int(round(y/s))
            x0 = clamp(x0,0,W-1); y0 = clamp(y0,0,H-1)
            x1 = clamp(x1,0,W-1); y1 = clamp(y1,0,H-1)
            rect_xyxy = (min(x0,x1), min(y0,y1), max(x0,x1), max(y0,y1))
        elif ev == cv2.EVENT_LBUTTONUP and drag is not None:
            x0,y0 = drag
            x1 = int(round(x/s)); y1 = int(round(y/s))
            x0 = clamp(x0,0,W-1); y0 = clamp(y0,0,H-1)
            x1 = clamp(x1,0,W-1); y1 = clamp(y1,0,H-1)
            rect_xyxy = (min(x0,x1), min(y0,y1), max(x0,y1), max(y0,y1))
            drag = None

    cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(title, min(max_w, W), min(max_h, H))
    cv2.setMouseCallback(title, on_mouse)

    accepted = False
    while True:
        s = scale()
        disp = draw(s)
        cv2.imshow(title, disp)
        k = cv2.waitKeyEx(20)
        if k in (13,10):  # ENTER
            if rect_xyxy is None: continue
            accepted = True; break
        if k == 27: break
        if k in (ord('r'), ord('R')): rect_xyxy = None

    cv2.destroyWindow(title)
    if not accepted or rect_xyxy is None: return None
    x1,y1,x2,y2 = rect_xyxy
    return (x1, y1, max(1, x2-x1+1), max(1, y2-y1+1))

# ───────────── Grid phase picker (click + nudge) ─────────────
def grid_phase_picker(frame_bgr, cell_w=CELL_SIZE, cell_h=CELL_SIZE_Y,
                      title="Click an intersection. W/S ±1 • . , ±0.1 (size) • Arrows/IJKL = phase. ENTER accept • R reset • ESC cancel",
                      max_w=1600, max_h=1000):
    H, W = frame_bgr.shape[:2]
    click = None
    cw, ch = float(cell_w), float(cell_h)
    # Start phases at 0; clicking sets absolute phase; then arrows/JL/IK nudge
    phase_x, phase_y = 0.0, 0.0

    def scale():
        try:
            _,_,ww,hh = cv2.getWindowImageRect(title)
            s = min(ww / max(1,W), hh / max(1,H))
            return s if np.isfinite(s) and s > 0 else min(1.0, max_w/W, max_h/H)
        except Exception:
            return min(1.0, max_w/W, max_h/H)

    def draw(s):
        disp = cv2.resize(frame_bgr, (max(1,int(W*s)), max(1,int(H*s))), interpolation=cv2.INTER_AREA)
        # draw grid from current phase_x/phase_y and cw/ch
        # verticals
        x = phase_x
        while x < W:
            cv2.line(disp, (int(round(x*s)), 0), (int(round(x*s)), int(H*s)-1), (0,0,255), max(1,int(1*s)))
            x += cw
        # horizontals
        y = phase_y
        while y < H:
            cv2.line(disp, (0, int(round(y*s))), (int(W*s)-1, int(round(y*s))), (0,0,255), max(1,int(1*s)))
            y += ch
        # click mark
        if click is not None:
            px, py = click
            cv2.circle(disp, (int(round(px*s)), int(round(py*s))), max(2,int(4*s)), (0,255,0), -1, cv2.LINE_AA)

        hud1 = f"Cell≈({ch:.2f},{cw:.2f})  W/S ±1, . , ±0.1"
        hud2 = f"Phase≈({phase_y:.2f},{phase_x:.2f})  Arrows ±1 (Y/X), I/K ±0.1 Y, J/L ±0.1 X"
        hud3 = "Click any grid intersection → sets phase; ENTER accept • R reset • ESC cancel"
        ytxt = max(24,int(24*s))
        for line in (hud1, hud2, hud3):
            cv2.putText(disp, line, (10, ytxt), cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.4,0.6*s), (0,255,255), max(1,int(2*s)), cv2.LINE_AA)
            ytxt += max(20,int(20*s))
        return disp

    def on_mouse(ev, x, y, flags, _):
        nonlocal click, phase_x, phase_y
        if ev == cv2.EVENT_LBUTTONDOWN:
            s = scale()
            px = x / max(1e-6, s); py = y / max(1e-6, s)
            if 0 <= px < W and 0 <= py < H:
                click = (float(px), float(py))
                # set absolute phase from click
                phase_x = float(px % cw)
                phase_y = float(py % ch)

    cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(title, min(max_w, W), min(max_h, H))
    cv2.setMouseCallback(title, on_mouse)

    accepted = False
    while True:
        s = scale()
        disp = draw(s)
        cv2.imshow(title, disp)
        k = cv2.waitKeyEx(20)
        if k in (13,10):  # ENTER
            if click is None: continue
            accepted = True; break
        if k == 27: break
        if k in (ord('r'), ord('R')):
            click = None
            phase_x, phase_y = 0.0, 0.0

        # size adjust
        if k in (ord('w'), ord('W')): cw += 1.0; ch += 1.0
        if k in (ord('s'), ord('S')): cw = max(0.1, cw-1.0); ch = max(0.1, ch-1.0)
        if k in (ord('.'), ord('>')): cw += 0.1; ch += 0.1
        if k in (ord(','), ord('<')): cw = max(0.1, cw-0.1); ch = max(0.1, ch-0.1)

        # phase adjust (±1 via arrows)
        if k == KEY_LEFT:  phase_x = (phase_x - 1.0) % cw
        if k == KEY_RIGHT: phase_x = (phase_x + 1.0) % cw
        if k == KEY_UP:    phase_y = (phase_y - 1.0) % ch
        if k == KEY_DOWN:  phase_y = (phase_y + 1.0) % ch
        # fine phase ±0.1 (I/K for Y, J/L for X)
        if k in (ord('j'), ord('J')): phase_x = (phase_x - 0.1) % cw
        if k in (ord('l'), ord('L')): phase_x = (phase_x + 0.1) % cw
        if k in (ord('i'), ord('I')): phase_y = (phase_y - 0.1) % ch
        if k in (ord('k'), ord('K')): phase_y = (phase_y + 0.1) % ch

    cv2.destroyWindow(title)
    if not accepted or click is None:
        return None
    return {"phase_x": float(phase_x), "phase_y": float(phase_y),
            "cell_w": float(cw), "cell_h": float(ch)}

# ───────────── tracking ─────────────
def make_edge_ring_template(img_bgr, box, ring=10):
    x,y,w,h = box
    roi = img_bgr[y:y+h, x:x+w]
    g = to_gray_f32(roi)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mask = np.zeros_like(mag, np.uint8)
    r = int(max(1, ring))
    mask[:r,:] = 1; mask[-r:,:] = 1; mask[:,:r] = 1; mask[:,-r:] = 1
    tmpl = mag * mask.astype(mag.dtype)
    m, s = cv2.meanStdDev(tmpl)
    if s[0,0] > 1e-6:
        tmpl = (tmpl - m[0,0]) / (s[0,0] + 1e-6)
    return tmpl

def track_window_next(frame_bgr, last_box, tmpl_edge_norm, search_margin=80, min_resp=0.25):
    H, W = frame_bgr.shape[:2]
    x,y,w,h = last_box
    sx0 = clamp(x - search_margin, 0, W-1)
    sy0 = clamp(y - search_margin, 0, H-1)
    sx1 = clamp(x + w + search_margin, 0, W)
    sy1 = clamp(y + h + search_margin, 0, H)
    patch = frame_bgr[sy0:sy1, sx0:sx1]
    g = to_gray_f32(patch)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    pm, ps = cv2.meanStdDev(mag)
    if ps[0,0] > 1e-6:
        mag = (mag - pm[0,0]) / (ps[0,0] + 1e-6)
    res = cv2.matchTemplate(mag, tmpl_edge_norm, cv2.TM_CCOEFF_NORMED)
    _, peak, _, loc = cv2.minMaxLoc(res)
    dx, dy = loc
    nx = int(sx0 + dx); ny = int(sy0 + dy)
    nx = clamp(nx, 0, W - w); ny = clamp(ny, 0, H - h)
    if peak < float(min_resp):  # keep previous if too weak
        return (x, y, w, h), float(peak)
    return (nx, ny, w, h), float(peak)

# ───────────── hole fill ─────────────
def fill_grow_sum_count(color_sum_f32, count_sum_f32, max_iters):
    if max_iters <= 0:
        return color_sum_f32, count_sum_f32
    if color_sum_f32.ndim == 2:
        color_sum_f32 = color_sum_f32[..., None]
    H, W, C = color_sum_f32.shape
    def box_sum_3(img2d):
        return cv2.boxFilter(img2d, -1, (3,3), normalize=False, borderType=cv2.BORDER_REFLECT101)
    zeros = (count_sum_f32 == 0)
    it = 0
    while np.any(zeros):
        if it >= max_iters:
            print(f"⚠️  Fill reached {max_iters} iters; stopping.")
            break
        sum_k = box_sum_3(count_sum_f32)
        sum_c = np.empty_like(color_sum_f32)
        for ch in range(C):
            sum_c[..., ch] = box_sum_3(color_sum_f32[..., ch])
        mask2 = zeros.astype(count_sum_f32.dtype)
        color_sum_f32 += sum_c * mask2[..., None]
        count_sum_f32 += sum_k * mask2
        new_zeros = (count_sum_f32 == 0)
        if np.array_equal(new_zeros, zeros):
            print("⚠️  Fill made no progress; aborting.")
            break
        zeros = new_zeros
        it += 1
    return color_sum_f32, count_sum_f32

# ───────────── main ─────────────
def main():
    if SAVE_DEBUG:
        ensure_dir(DEBUG_DIR)
        ensure_dir(os.path.join(DEBUG_DIR, "overlays"))
        log_path = os.path.join(DEBUG_DIR, "tracking_log.csv")
        log_f = open(log_path, "w", newline="")
        logger = csv.writer(log_f)
        logger.writerow(["frame_idx","x","y","w","h","match_response","samples_this_frame"])
    else:
        logger = None
        log_f = None

    cap0 = cv2.VideoCapture(VIDEO_PATH)
    if not cap0.isOpened():
        raise SystemExit(f"❌ Could not open video: {VIDEO_PATH}")
    if START_AT_FRAME > 0:
        cap0.set(cv2.CAP_PROP_POS_FRAMES, START_AT_FRAME)
    ok, frame0 = cap0.read()
    cap0.release()
    if not ok:
        raise SystemExit("❌ Could not read start frame.")

    # Step 1: pick the moving window
    print("🖼️  Pick the moving window…")
    init_box = window_picker(frame0)
    if init_box is None:
        raise SystemExit("Canceled window pick.")
    x0, y0, w0, h0 = init_box

    # Step 2: set grid size & phase
    print("➕ Click ONE stationary grid intersection and align the grid with keys…")
    phase_info = grid_phase_picker(frame0, cell_w=CELL_SIZE, cell_h=CELL_SIZE_Y)
    if phase_info is None:
        raise SystemExit("Canceled grid phase pick.")
    phase_x = phase_info["phase_x"]
    phase_y = phase_info["phase_y"]
    cell_w  = phase_info["cell_w"]
    cell_h  = phase_info["cell_h"]

    print(f"✅ Window {init_box},  Grid cell≈({cell_h:.3f},{cell_w:.3f}),  Phase≈({phase_y:.3f},{phase_x:.3f})")

    tmpl = make_edge_ring_template(frame0, init_box, ring=TRACK_EDGE_RING)

    # Prepare SR canvas (window-relative)
    W_sr = int(round(w0 * SR_FACTOR))
    H_sr = int(round(h0 * SR_FACTOR))
    C = frame0.shape[2] if frame0.ndim == 3 else 1
    color_sum = np.zeros((H_sr, W_sr, C), dtype=np.float32)
    count_sum = np.zeros((H_sr, W_sr), dtype=np.float32)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit(f"❌ Could not re-open video: {VIDEO_PATH}")
    if START_AT_FRAME > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, START_AT_FRAME)

    last_box = (x0, y0, w0, h0)
    processed = 0
    fidx = START_AT_FRAME - 1

    print("🔄 Tracking & projecting…")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fidx += 1
        if (fidx - START_AT_FRAME) % FRAME_STRIDE != 0:
            continue
        if processed >= MAX_FRAMES:
            break

        # Track window
        box, resp = track_window_next(frame, last_box, tmpl,
                                      search_margin=TRACK_SEARCH_MARGIN,
                                      min_resp=TRACK_MIN_RESPONSE)
        last_box = box
        x, y, w, h = box

        # Build stationary **centre** grid inside this window
        def first_center_k_at_or_after(a0, phase, step):
            # smallest integer k such that phase + (k + 0.5)*step >= a0
            return math.ceil(((a0 - phase) / step) - 0.5)

        kx0 = first_center_k_at_or_after(x, phase_x, cell_w)
        ky0 = first_center_k_at_or_after(y, phase_y, cell_h)

        xs = []
        k = kx0
        while True:
            cx = phase_x + (k + 0.5) * cell_w
            if cx >= x + w: break
            if cx >= x: xs.append(cx)
            k += 1

        ys = []
        k = ky0
        while True:
            cy = phase_y + (k + 0.5) * cell_h
            if cy >= y + h: break
            if cy >= y: ys.append(cy)
            k += 1

        samples_this = 0
        if xs and ys:
            Xg, Yg = np.meshgrid(np.array(xs, np.float64),
                                 np.array(ys, np.float64), indexing="xy")
            Xi = np.rint(Xg).astype(int).clip(0, frame.shape[1]-1)
            Yi = np.rint(Yg).astype(int).clip(0, frame.shape[0]-1)
            vals = frame[Yi, Xi].astype(np.float32)  # (Ny,Nx,3)

            # Project into canonical window coordinates
            Xw = (Xg - x) * SR_FACTOR
            Yw = (Yg - y) * SR_FACTOR

            Hsr, Wsr = color_sum.shape[:2]
            if SR_SPLAT_MODE.lower() == "nearest":
                Xn = np.rint(Xw).astype(np.int32)
                Yn = np.rint(Yw).astype(np.int32)
                valid = (Xn >= 0) & (Xn < Wsr) & (Yn >= 0) & (Yn < Hsr)
                if np.any(valid):
                    xi = Xn[valid].ravel(); yi = Yn[valid].ravel()
                    v  = vals[valid].reshape(-1, C)
                    idx = yi * Wsr + xi
                    cs = color_sum.reshape(-1, C)
                    np.add.at(cs, idx, v)
                    np.add.at(count_sum.ravel(), idx, 1.0)
                    samples_this = int(valid.sum())
            else:
                x0i = np.floor(Xw).astype(np.int32)
                y0i = np.floor(Yw).astype(np.int32)
                wx = (Xw - x0i).astype(np.float32)
                wy = (Yw - y0i).astype(np.float32)
                v  = vals.reshape(-1, C)
                samples_this = Xw.size
                coords = [
                    (x0i,     y0i,     (1 - wx) * (1 - wy)),
                    (x0i + 1, y0i,     wx       * (1 - wy)),
                    (x0i,     y0i + 1, (1 - wx) * wy),
                    (x0i + 1, y0i + 1, wx       * wy),
                ]
                for Xn, Yn, Wn in coords:
                    valid = (Xn >= 0) & (Xn < Wsr) & (Yn >= 0) & (Yn < Hsr) & (Wn > 0)
                    if not np.any(valid): continue
                    xi = Xn[valid].ravel(); yi = Yn[valid].ravel()
                    wgt = Wn[valid].ravel().astype(np.float32)
                    idx = yi * Wsr + xi
                    cs = color_sum.reshape(-1, C)
                    np.add.at(cs, idx, v[valid.reshape(-1)] * wgt[:, None])
                    np.add.at(count_sum.ravel(), idx, wgt)

        processed += 1

        # Debug: CSV + overlays
        if logger is not None:
            logger.writerow([fidx, x, y, w, h, f"{resp:.4f}", samples_this])
            log_f.flush()
        if SAVE_DEBUG:
            overlay_this = SAVE_OVERLAY_EVERY_FRAME or (OVERLAY_PERIOD>0 and processed % OVERLAY_PERIOD==0)
            if overlay_this:
                vis = frame.copy()
                draw_window((x,y,w,h), vis, (0,255,255), 2)
                # draw grid lines (for visual reference)
                # verticals
                xv = phase_x
                while xv < frame.shape[1]:
                    cv2.line(vis, (int(round(xv)), 0), (int(round(xv)), frame.shape[0]-1),
                             (0,0,255), 1, cv2.LINE_AA)
                    xv += cell_w
                # horizontals
                yv = phase_y
                while yv < frame.shape[0]:
                    cv2.line(vis, (0, int(round(yv))), (frame.shape[1]-1, int(round(yv))),
                             (0,0,255), 1, cv2.LINE_AA)
                    yv += cell_h
                # centres inside the window
                if OVERLAY_DRAW_CENTRES and xs and ys:
                    for xg in xs:
                        for yg in ys:
                            cv2.circle(vis, (int(round(xg)), int(round(yg))), 1, (0,255,0), -1, cv2.LINE_AA)
                # HUD
                cv2.putText(vis, f"f={fidx}  resp={resp:.3f}  samples={samples_this}  "
                                  f"cell=({cell_h:.2f},{cell_w:.2f})  phase=({phase_y:.2f},{phase_x:.2f})",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
                cv2.imwrite(os.path.join(DEBUG_DIR, "overlays", f"grid_{fidx:06}.png"), vis)

    cap.release()
    if SAVE_DEBUG and log_f:
        log_f.close()

    if count_sum.sum() == 0:
        print("❌ No samples collected; check the picks or parameters.")
        return

    recon_before = (color_sum / np.maximum(count_sum, 1e-6)[..., None]).clip(0,255).astype(np.uint8)
    cv2.imwrite("reconstruction_sr_before_fill.png", recon_before)
    print("✅ wrote reconstruction_sr_before_fill.png")

    if FILL_MAX_ITERS > 0 and np.any(count_sum == 0):
        print("🧩 Filling holes (grow)…")
        csum_filled, ksum_filled = fill_grow_sum_count(color_sum.copy(), count_sum.copy(), FILL_MAX_ITERS)
        recon = (csum_filled / np.maximum(ksum_filled, 1e-6)[..., None]).clip(0,255).astype(np.uint8)
    else:
        recon = recon_before

    cv2.imwrite("reconstruction_sr.png", recon)
    print(f"✅ wrote reconstruction_sr.png (SR_FACTOR={SR_FACTOR}, frames_used={processed})")


if __name__ == "__main__":
    main()
