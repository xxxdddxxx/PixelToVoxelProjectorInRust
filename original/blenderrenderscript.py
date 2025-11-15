#!/usr/bin/env python3
"""
blender_multi_camera_render.py
──────────────────────────────
Creates many cameras, renders every camera–frame pair, and writes a
metadata-rich JSON file that can **resume safely** after a crash.

Key ideas
---------
* NEW_PHOTOS = True  ➜ we generate every camera pose up-front, write a
  JSON skeleton with `"rendered": false` for every image, then render.
* NEW_PHOTOS = False ➜ we open the existing JSON, skip items already
  marked `"rendered": true`, and continue where we left off.

Any time an image finishes rendering we **immediately** flip its flag to
true and flush the whole JSON back to disk, so at worst you lose *one*
image on a hard Blender crash.

The companion C++ program now ignores entries whose `"rendered"` flag is
missing or false (see §2).
"""
# ─────────────────────────────────────────────────────────────────────────────
# 0) Imports & GLOBAL SETTINGS  ───────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
import bpy, os, json, random, math
from mathutils import Vector, Euler

# === USER-TWEAKABLE CONSTANTS =================================================
TEST_MOVE = False
NEW_PHOTOS          = True                     # • flip to False to resume
OUTPUT_DIR          = r"yourpath"
RES_X, RES_Y        = 1920, 1080              # render resolution
N_CAMERAS           = 3000
if TEST_MOVE :
    N_CAMERAS = 1
SPEED_RANGE         = (-0.00, 0.00)
TARGET_POINT        = (1, 1, 14000)      # world-space XYZ the cams must see
POS_RANGE           = (-25000, 25000)       # uniform cube for camera XYZ
ANG_RANGE           = (-360, 360)             # yaw/pitch/roll range (deg)
FOV_RANGE           = (50, 70)                # horizontal FOV range (deg)
OBJ_NAME            = "bud"                   # main object being filmed
NOISE_SOURCE_NAME   = "tru"                   # mesh duplicated for clutter
NOISE_PER_CAM       = 0
META_PATH           = os.path.join(OUTPUT_DIR, "metadata.json")
IMAGE_FMT           = 'PNG'                   # PNG keeps every pixel
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# 1)  Utility helpers (unchanged or slightly tweaked)  ────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def ensure_camera_exists(name: str):
    """Get or create a Blender camera object named *name*."""
    if name in bpy.data.objects:
        return bpy.data.objects[name]
    cam_data = bpy.data.cameras.new(name)
    cam_obj  = bpy.data.objects.new(name, cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    return cam_obj


def random_camera_pose(position_range, angle_range, fov_range):
    """Return dict with random position, yaw, pitch, roll, fov (deg)."""
    x = random.uniform(*position_range)
    y = random.uniform(*position_range)
    z = random.uniform(*(-30, 40))
    return dict(position=(x, y, z),
                yaw   = random.uniform(*angle_range),
                pitch = random.uniform(*angle_range),
                roll  = random.uniform(*angle_range),
                fov   = random.uniform(*fov_range))


def is_point_visible(cam_info, point_xyz, aspect_ratio=RES_X/RES_Y):
    """
    Cheap test: does *point_xyz* fall inside the camera's view frustum?
    Horizontal FOV is stored in cam_info["fov"].
    """
    # world → camera space
    R_wc = Euler((math.radians(cam_info["pitch"]),
                  math.radians(cam_info["roll"]),
                  math.radians(cam_info["yaw"])), 'XYZ').to_matrix()
    vec_cam = R_wc.transposed() @ (Vector(point_xyz) - Vector(cam_info["position"]))
    if vec_cam.z >= 0:                               # camera looks down −Z
        return False
    
    if -20000 < cam_info["position"][0] < 20000 and -20000 < cam_info["position"][1] < 20000:# and 10 < cam_info["position"][2] < 50:
        return False
    
    h_fov  = math.radians(cam_info["fov"])
    v_fov  = 2 * math.atan(math.tan(h_fov/2) / aspect_ratio)
    half_w = -vec_cam.z * math.tan(h_fov/2)
    half_h = -vec_cam.z * math.tan(v_fov/2)
    return abs(vec_cam.x) <= half_w and abs(vec_cam.y) <= half_h


def random_visible_camera_pose(target_point):
    """Try repeatedly until the target is inside the frustum."""
    for _ in range(10000):
        pose = random_camera_pose(POS_RANGE, ANG_RANGE, FOV_RANGE)
        if is_point_visible(pose, target_point):
            return pose
    raise RuntimeError("Could not find a valid camera pose.")


def apply_camera_pose(cam_obj, cam_info):
    """Push dict ➜ actual Blender camera transform."""
    cam_obj.location        = cam_info["position"]
    cam_obj.rotation_mode   = 'XYZ'
    cam_obj.rotation_euler  = (math.radians(cam_info["pitch"]),
                               math.radians(cam_info["roll"]),
                               math.radians(cam_info["yaw"]))
    cam_obj.data.angle      = math.radians(cam_info["fov"])


def set_render_distance(camera_obj, clip_start=0.1, clip_end=3e7):
    cam = camera_obj
    cam.data.clip_start = clip_start
    cam.data.clip_end   = clip_end


def place_noise_objects(base_obj_name, camera_obj, num_objects,
                        min_dist, max_dist, lateral_spread, vertical_spread):
    """Duplicate *base_obj_name* around the camera FOV to add clutter."""
    if base_obj_name not in bpy.data.objects:
        return []
    base    = bpy.data.objects[base_obj_name]
    fwd     = (camera_obj.matrix_world.to_quaternion() @ Vector((0, 0, -1))).normalized()
    right   = (camera_obj.matrix_world.to_quaternion() @ Vector((1, 0, 0))).normalized()
    up      = (camera_obj.matrix_world.to_quaternion() @ Vector((0, 1, 0))).normalized()
    objs    = []

    for _ in range(num_objects):
        dup  = base.copy(); dup.data = base.data.copy()
        bpy.context.scene.collection.objects.link(dup)
        dist = random.uniform(min_dist, max_dist)
        dup.location  = (camera_obj.location + fwd * dist
                         + right * random.uniform(-lateral_spread,  lateral_spread)
                         +   up * random.uniform(-vertical_spread, vertical_spread))
        dup.rotation_euler = (random.random()*2*math.pi,
                              random.random()*2*math.pi,
                              random.random()*2*math.pi)
        s = random.uniform(1, 1.5); dup.scale = (s, s, s)
        objs.append(dup)
    return objs


def remove_objects(objs):
    for ob in objs:
        if ob.name in bpy.data.objects:
            for c in ob.users_collection:
                c.objects.unlink(ob)
            bpy.data.objects.remove(ob, do_unlink=True)


def setup_render(out_dir, res_x, res_y, fmt):
    os.makedirs(out_dir, exist_ok=True)
    scn = bpy.context.scene
    scn.render.resolution_x       = res_x
    scn.render.resolution_y       = res_y
    scn.render.resolution_percentage = 100
    scn.render.image_settings.file_format = fmt



    
    return out_dir


def render_image(cam, cam_idx, frame_idx, out_dir):
    bpy.context.scene.camera = cam
    fstem  = f"image_{cam_idx:03d}_frame_{frame_idx:03d}"
    bpy.context.scene.render.filepath = os.path.join(out_dir, fstem)
    bpy.ops.render.render(write_still=True)
    return fstem + ".png"   # relative file name for JSON


# ─────────────────────────────────────────────────────────────────────────────
# 2)  JSON helpers – load, save, flush  ───────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def dump_metadata(meta, path):
    """Write *meta* (list of dict) atomically to *path*."""
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(meta, fh, indent=2)
    os.replace(tmp, path)   # atomic on most filesystems


def load_metadata(path):
    with open(path, "r") as fh:
        return json.load(fh)


# ─────────────────────────────────────────────────────────────────────────────
# 3)  Main program  ───────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def main():
    
    global NEW_PHOTOS          # so we can flip it dynamically

    # If we *meant* to resume but no file exists yet → switch to fresh run
    if not NEW_PHOTOS and not os.path.exists(META_PATH):
        print("[INFO] metadata.json not found – switching NEW_PHOTOS to True")
        NEW_PHOTOS = True
    # sanity checks -----------------------------------------------------------
    if OBJ_NAME not in bpy.data.objects:
        raise RuntimeError(f"No object named '{OBJ_NAME}' in the scene.")
    main_obj = bpy.data.objects[OBJ_NAME]

    # Path the object moves along (unchanged from your example) --------------
    object_path = [(x*5, 1, 14000) for x in range(8)]     # 12 frames
    if TEST_MOVE:
        object_path = [(x*0, 1, 000) for x in range(100)]     # 12 frames

    # Render setup -----------------------------------------------------------
    setup_render(OUTPUT_DIR, RES_X, RES_Y, IMAGE_FMT)
    scn = bpy.context.scene
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.get_devices()
    prefs.compute_device_type = 'CUDA'      # or OPTIX / HIP / METAL
    scn.cycles.device          = 'GPU'
    scn.render.use_persistent_data = True   # ★ keeps kernels/BVH in RAM
    scn.cycles.use_adaptive_sampling = True # optional

    # THIS is the one that matters for your slowdown
    scn.render.use_persistent_data = True   # <-- instead of scn.cycles.*

    # Either create a fresh metadata skeleton or reopen the old one ----------
    if NEW_PHOTOS:
        camera_infos = [random_visible_camera_pose(TARGET_POINT)
                        for _ in range(N_CAMERAS)]
        metadata = []
        for f_idx, obj_pos in enumerate(object_path):
            for c_idx, info in enumerate(camera_infos):
                print(f_idx)
                #info["position"][0] += 5 * f_idx
                added = list((info["position"][0] + (random.uniform(*SPEED_RANGE) * f_idx), info["position"][1] + (random.uniform(*SPEED_RANGE) * f_idx), 0.2))
                metadata.append(dict(camera_index=c_idx,
                                     frame_index =f_idx,
                                     camera_position = added,#list(info["position"]),
                                     yaw         = info["yaw"],
                                     pitch       = info["pitch"],
                                     roll        = info["roll"],
                                     fov_degrees = info["fov"],
                                     object_name = OBJ_NAME,
                                     object_location = list(obj_pos),
                                     image_file  = "",          # filled later
                                     rendered    = False))      # !! key flag
                                     
                print(list(info["position"]))
#                print(camera_position)
        dump_metadata(metadata, META_PATH)
        print(f"[INIT] JSON skeleton with {len(metadata)} images written → {META_PATH}")
    else:
        metadata = load_metadata(META_PATH)
        # cameras may already exist; if not, create in loop below
        camera_infos = None                                   # read per entry
    
    #print(list(info["position"]))
    
    # Quick index: (cam_idx) ➜ Blender camera object -------------------------
    cam_objs = {}

    # Main render loop --------------------------------------------------------
    total   = len(metadata)
    pending = sum(not m["rendered"] for m in metadata)
    print(f"[START] {pending}/{total} images still to render.")
    SAVE_EVERY = 50          # tweak: 1 = every frame, bigger = faster, riskier
    save_counter = 0
    for entry in metadata:
        if entry["rendered"]:
            continue                                  # skip completed images
        save_counter += 1

        # Lazily create / fetch the camera object ----------------------------
        cam_idx = entry["camera_index"]
        if cam_idx not in cam_objs:
            cam_objs[cam_idx] = ensure_camera_exists(f"Camera_{cam_idx:03d}")
        cam = cam_objs[cam_idx]

        # Feed pose into the camera (info always stored in entry) ------------
        apply_camera_pose(cam, dict(position=entry["camera_position"],
                                    yaw   =entry["yaw"],
                                    pitch =entry["pitch"],
                                    roll  =entry["roll"],
                                    fov   =entry["fov_degrees"]))
        set_render_distance(cam)

        # Move main object to correct path location --------------------------
        main_obj.location = entry["object_location"]

        # Add/-remove scene clutter ------------------------------------------
        noise = place_noise_objects(NOISE_SOURCE_NAME, cam, NOISE_PER_CAM,
                                    1_000, 14_000, 5_000, 5_000)

        # Render -------------------------------------------------------------
        rel_path = render_image(cam, cam_idx, entry["frame_index"], OUTPUT_DIR)
        entry["image_file"] = rel_path
        entry["rendered"]   = True
        if save_counter >= SAVE_EVERY:
            dump_metadata(metadata, META_PATH)
            save_counter = 0
        remove_objects(noise)

        pending -= 1
        print(f"[OK]  cam {cam_idx:03d} frame {entry['frame_index']:03d} "
              f"→ {rel_path}   ({pending} left)")

    print("[DONE] All images rendered and metadata finalised.")


if __name__ == "__main__":
    main()
