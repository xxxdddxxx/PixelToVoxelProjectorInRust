/*
  This code is a direct and optimized translation
  of the pixel to voxel projection from ConsistentlyInconsistentYT.

  There have been added a few optimizations like Atomic

  I am not very experienced rust coder so i decided to develop my skills
  by translating this project in rust.
*/

use rand::thread_rng;
use stb_image::image::LoadResult;
use stb_image::image::load_with_depth;
use rand::rngs::ThreadRng;
use serde_json::Value;
use std::fs::File;
use std::collections::HashMap;
use std::io::{BufReader, BufWriter, Write};
use std::time::Instant;
use rand::Rng;
use std::arch::x86_64::*;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};
use std::mem;

#[derive(Default)]
#[repr(align(4))]
struct AtomicF32 {
    v: AtomicU32,
}

impl AtomicF32 {
    fn new(value: f32) -> Self {
        let u = value.to_bits();
        AtomicF32 { v: AtomicU32::new(u) }
    }

    fn fetch_add(&self, value: f32, order: Ordering) {
        let mut current_f = self.load(order);
        loop {
            let new_f = current_f + value;
            let current_u = current_f.to_bits();
            let new_u = new_f.to_bits();
            match self.v.compare_exchange_weak(current_u, new_u, order, order) {
                Ok(_) => return,
                Err(e) => {
                    current_f = f32::from_bits(e);
                }
            }
        }
    }

    fn load(&self, order: Ordering) -> f32 {
        f32::from_bits(self.v.load(order))
    }

    fn store(&self, value: f32, order: Ordering) {
        self.v.store(value.to_bits(), order)
    }
}

#[derive(Default, Clone)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32
}

struct Mat3 {
    m: [f32; 9],
}

#[derive(Default, Clone)]
struct FrameInfo {
    camera_index: i32,
    frame_index: i32,
    camera_position: Vec3,
    yaw: f32,
    pitch: f32,
    roll: f32,
    fov_degrees: f32,
    image_file: String,
}
#[derive(Default)]
pub struct ImageGray {
    width: i32,
    height: i32,
    pixels: Vec<f32>
}

struct RayStep {
    ix: i32,
    iy: i32,
    iz: i32,
    step_count: i32,
    distance: f32,
}

struct MotionMask {
    width: i32,
    height: i32,
    changed: Vec<bool>,
    diff: Vec<f32>,
}

/*
    the functions that the code specified
*/
fn deg2rad(deg: f32) -> f32 {
    deg * std::f32::consts::PI / 180.0
}

fn normalize(v: Vec3) -> Vec3 {
    let len = (v.x * v.x + v.y * v.y + v.z * v.z).sqrt();
    if len < 1e-12 {
        return Vec3 { x: 0.0, y: 0.0, z: 0.0 };
    }
    let inv_len = 1.0 / len;
    Vec3 {
        x: v.x * inv_len,
        y: v.y * inv_len,
        z: v.z * inv_len
    }
}


fn mat3_mul_vec3(M: &Mat3, v: Vec3) -> Vec3 {
    let mut r = Vec3{x: 0.0, y: 0.0, z: 0.0};
    r.x = M.m[0]*v.x + M.m[1]*v.y + M.m[2]*v.z;
    r.y = M.m[3]*v.x + M.m[4]*v.y + M.m[5]*v.z;
    r.z = M.m[6]*v.x + M.m[7]*v.y + M.m[8]*v.z;
    r
}

fn matmul3x3(A: [f32; 9], B: [f32; 9]) -> [f32; 9] {
    let mut C: [f32; 9] = [0.0; 9];
    for row in 0..3 {
        for col in 0..3 {
            C[row*3+col] =  A[row*3  ]*B[0*3+col] +
                A[row*3+1]*B[1*3+col] +
                A[row*3+2]*B[2*3+col];
        }
    }
    C
}

fn rotation_matrix_yaw_pitch_roll(yaw_deg: f32, pitch_deg: f32, roll_deg: f32) -> Mat3 {
    let y = deg2rad(yaw_deg);
    let p = deg2rad(pitch_deg);
    let r = deg2rad(roll_deg);

    let cy = f32::cos(y);
    let sy = f32::sin(y);
    let cr = f32::cos(r);
    let sr = f32::sin(r);
    let cp = f32::cos(p);
    let sp = f32::sin(p);

    let rz = [
        cy, -sy, 0.0,
        sy, cy, 0.0,
        0.0, 0.0, 1.0
    ];

    let ry = [
        cr, 0.0, sr,
        0.0, 1.0, 0.0,
        -sr, 0.0, cr
    ];

    let rx = [
        1.0, 0.0, 0.0,
        0.0, cp, -sp,
        0.0, sp, cp
    ];

    let mut out = Mat3 { m: [0.0; 9] };

    out.m = matmul3x3(rz, ry);
    out.m = matmul3x3(out.m, rx);

    out
}

fn load_image_gray(img_path: &str, out: &mut ImageGray) -> bool {
    let img = match load_with_depth(img_path, 1, false) {
        LoadResult::ImageU8(img) => img,
        _ => return false,
    };

    let width = img.width as i32;
    let height = img.height as i32;
    let size = (width * height) as usize;

    out.width = width;
    out.height = height;
    out.pixels = vec![0.0; size];

    let mut rng = thread_rng();
    let mut idx = 0;

    unsafe {
        while idx + 4 <= size {
            let original_values = [
                img.data[idx] as f32,
                img.data[idx + 1] as f32,
                img.data[idx + 2] as f32,
                img.data[idx + 3] as f32,
            ];

            let noise_values = [
                rng.gen_range(-1.0f32..=1.0f32),
                rng.gen_range(-1.0f32..=1.0f32),
                rng.gen_range(-1.0f32..=1.0f32),
                rng.gen_range(-1.0f32..=1.0f32),
            ];

            for i in 0..4 {
                let noisy_value = (original_values[i] + noise_values[i]).clamp(0.0, 255.0);
                out.pixels[idx + i] = noisy_value;
            }
            idx += 4;
        }
    }

    while idx < size {
        let original_value = img.data[idx] as f32;
        let noise: f32 = rng.gen_range(-1.0f32..=1.0f32);
        out.pixels[idx] = (original_value + noise).clamp(0.0, 255.0);
        idx += 1;
    }


    true
}
fn detect_motion(prev: &ImageGray, next: &ImageGray, threshold: f32) -> MotionMask {
    let mut mm: MotionMask = MotionMask {
        width: prev.width,
        height: prev.height,
        changed: Vec::new(),
        diff: Vec::new(),
    };

    if prev.width != next.width || prev.height != next.height {
        eprintln!("[detect_motion][-] Images differ in size. Can't do motion detection!");
        mm.width = 0;
        mm.height = 0;
        return mm;
    }

    let size = (prev.width * prev.height) as usize;

    mm.changed.reserve(size);
    mm.diff.reserve(size);

    mm.changed.resize(size, false);
    mm.diff.resize(size, 0.0);

    for i in 0..size {
        let d = (prev.pixels[i] - next.pixels[i]).abs();
        mm.diff[i] = d;
        mm.changed[i] = d > threshold;
    }

    mm
}

/*
    for consistency i specified the safe_div function
    because there is no native safe_div in rust
*/
fn safe_div(a: f32, b: f32) -> f32 {
    if b.abs() < 1e-12 { f32::INFINITY } else { a / b }
}

fn load_metadata(json_path: &str) -> Vec<FrameInfo> {
    let file = match File::open(json_path) {
        Ok(f) => f,
        Err(_) => {
            eprintln!("[-][load_metadata] Cannot open {}", json_path);
            return Vec::new();
        }
    };

    let reader = BufReader::new(file);
    let json: Value = match serde_json::from_reader(reader) {
        Ok(j) => j,
        Err(_) => {
            eprintln!("[-][load_metadata] Failed to parse JSON.");
            return Vec::new();
        }
    };

    if !json.is_array() {
        eprintln!("[-][load_metadata] JSON top level is not an array.");
        return Vec::new();
    }

    let mut frames = Vec::new();

    for entry in json.as_array().unwrap() {
        let mut fi = FrameInfo::default();
        fi.camera_index = entry.get("camera_index").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
        fi.frame_index  = entry.get("frame_index").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
        fi.yaw          = entry.get("yaw").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        fi.pitch        = entry.get("pitch").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        fi.roll         = entry.get("roll").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        fi.fov_degrees  = entry.get("fov_degrees").and_then(|v| v.as_f64()).unwrap_or(60.0) as f32;
        fi.image_file   = entry.get("image_file").and_then(|v| v.as_str()).unwrap_or("").to_string();

        if let Some(pos_arr) = entry.get("camera_position").and_then(|v| v.as_array()) {
            if pos_arr.len() >= 3 {
                fi.camera_position.x = pos_arr[0].as_f64().unwrap_or(0.0) as f32;
                fi.camera_position.y = pos_arr[1].as_f64().unwrap_or(0.0) as f32;
                fi.camera_position.z = pos_arr[2].as_f64().unwrap_or(0.0) as f32;
            }
        }

        frames.push(fi);
    }

    frames
}
fn cast_ray_into_grid(
    camera_pos: Vec3,
    dir_normalized: Vec3,
    N: i32,
    voxel_size: f32,
    grid_center: Vec3) -> Vec<RayStep> {
    let mut steps: Vec<RayStep> = vec![];
    steps.reserve(64);

    let half_size: f32 = 0.5 * (N as f32 * voxel_size);
    let grid_min: Vec3 = Vec3{
        x: grid_center.x - half_size,
        y: grid_center.y - half_size,
        z: grid_center.z - half_size
    };

    let grid_max: Vec3 = Vec3{
        x: grid_center.x + half_size,
        y: grid_center.y + half_size,
        z: grid_center.z + half_size
    };

    let mut t_min: f32 = 0.0;
    let mut t_max: f32 = f32::INFINITY;

    for i in 0..3 {
        let origin = if i == 0 { camera_pos.x } else if i == 1 { camera_pos.y } else { camera_pos.z };
        let d = if i == 0 { dir_normalized.x } else if i == 1 { dir_normalized.y } else { dir_normalized.z };
        let mn = if i == 0 { grid_min.x } else if i == 1 { grid_min.y } else { grid_min.z };
        let mx = if i == 0 { grid_max.x } else if i == 1 { grid_max.y } else { grid_max.z };

        if d.abs() < 1e-12 {
            if origin < mn || origin > mx {
                return steps;
            }
        } else {
            let t1 = (mn - origin) / d;
            let t2 = (mx - origin) / d;
            let t_near = t1.min(t2);
            let t_far = t1.max(t2);
            if t_near > t_min { t_min = t_near; }
            if t_far < t_max { t_max = t_far; }
            if t_min > t_max {
                return steps;
            }
        }
    }

    if t_min < 0.0 {
        t_min = 0.0;
    }

    let start_world: Vec3 = Vec3{
        x: camera_pos.x + t_min*dir_normalized.x,
        y: camera_pos.y + t_min*dir_normalized.y,
        z: camera_pos.z + t_min*dir_normalized.z
    };

    let fx: f32 = (start_world.x - grid_min.x) / voxel_size;
    let fy: f32 = (start_world.y - grid_min.y) / voxel_size;
    let fz: f32 = (start_world.z - grid_min.z) / voxel_size;

    let mut ix: i32 = fx as i32;
    let mut iy: i32 = fy as i32;
    let mut iz: i32 = fz as i32;

    if ix<0 || ix>=N || iy<0 || iy>=N || iz<0 || iz>=N {
        return steps;
    }

    let step_x: i32 = if dir_normalized.x >= 0.0 { 1 } else { -1 };
    let step_y: i32 = if dir_normalized.y >= 0.0 { 1 } else { -1 };
    let step_z: i32 = if dir_normalized.z >= 0.0 { 1 } else { -1 };

    let boundary_in_world_x = |i_x: i32| grid_min.x + i_x as f32 * voxel_size;
    let boundary_in_world_y = |i_y: i32| grid_min.y + i_y as f32 * voxel_size;
    let boundary_in_world_z = |i_z: i32| grid_min.z + i_z as f32 * voxel_size;

    let nx_x = ix + if step_x > 0 { 1 } else { 0 };
    let nx_y = iy + if step_y > 0 { 1 } else { 0 };
    let nx_z = iz + if step_z > 0 { 1 } else { 0 };

    let next_bx = boundary_in_world_x(nx_x);
    let next_by = boundary_in_world_y(nx_y);
    let next_bz = boundary_in_world_z(nx_z);

    let mut t_max_x = safe_div(next_bx - camera_pos.x, dir_normalized.x);
    let mut t_max_y = safe_div(next_by - camera_pos.y, dir_normalized.y);
    let mut t_max_z = safe_div(next_bz - camera_pos.z, dir_normalized.z);

    let t_delta_x = safe_div(voxel_size, dir_normalized.x.abs());
    let t_delta_y = safe_div(voxel_size, dir_normalized.y.abs());
    let t_delta_z = safe_div(voxel_size, dir_normalized.z.abs());

    let mut t_current = t_min;
    let mut step_count = 0;

    if t_max_x < t_min { t_max_x = t_min + t_delta_x; }
    if t_max_y < t_min { t_max_y = t_min + t_delta_y; }
    if t_max_z < t_min { t_max_z = t_min + t_delta_z; }

    while t_current <= t_max {
        let mut rs: RayStep = RayStep{
            ix: 0,
            iy: 0,
            iz: 0,
            step_count: 0,
            distance: 0.0

        };
        rs.ix = ix;
        rs.iy = iy;
        rs.iz = iz;
        rs.step_count = step_count;
        rs.distance = t_current;

        steps.push(rs);

        let t_next = t_max_x.min(t_max_y.min(t_max_z)).min(t_max);

        if t_next == t_max {
            break;
        }

        t_current = t_next;

        if t_next == t_max_x {
            ix += step_x;
            t_max_x += t_delta_x;
        }
        if t_next == t_max_y {
            iy += step_y;
            t_max_y += t_delta_y;
        }
        if t_next == t_max_z {
            iz += step_z;
            t_max_z += t_delta_z;
        }

        step_count += 1;
        if ix<0 || ix>=N || iy<0 || iy>=N || iz<0 || iz>=N {
            break;
        }
    }

    steps
}


fn main() -> std::io::Result<()> {
    let start_total = Instant::now();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: {} <metadata.json> <image_folder> <output_voxel_bin>", args[0]);
        return Ok(());
    }

    let metadata_path = &args[1];
    let images_folder = &args[2];
    let output_bin = &args[3];

    let start_load = Instant::now();
    println!("[*] Loading metadata from {}", metadata_path);
    let frames = load_metadata(metadata_path);
    if frames.is_empty() {
        eprintln!("[-] No frames loaded.");
        return Ok(());
    }
    println!("[+] Loaded {} frames in {:?}", frames.len(), start_load.elapsed());

    let mut frames_by_cam: HashMap<i32, Vec<FrameInfo>> = HashMap::new();
    for f in &frames {
        frames_by_cam.entry(f.camera_index).or_default().push(f.clone());
    }

    for (cam_id, frames) in frames_by_cam.iter_mut() {
        frames.sort_by_key(|a| a.frame_index);
        println!("[+] Camera {} has {} frames.", cam_id, frames.len());
    }

    const N: usize = 500;
    const VOXEL_SIZE: f32 = 6.0;
    let grid_center = Vec3 { x: 0.0, y: 0.0, z: 500.0 };

    let voxel_grid: Vec<AtomicF32> = (0..N * N * N).map(|_| AtomicF32::new(0.0)).collect();

    let voxel_grid_ref = &voxel_grid;

    println!("[+] Initialized voxel grid with size {}x{}x{}", N, N, N);

    let motion_threshold = 2.0f32;
    let mut total_processed_pixels = AtomicU32::new(0);

    let start_processing = Instant::now();

    let mut frame_pairs: Vec<(FrameInfo, FrameInfo)> = Vec::new();

    for cam_frames in frames_by_cam.values() {
        if cam_frames.len() < 2 { continue; }
        for i in 1..cam_frames.len() {
            frame_pairs.push((cam_frames[i-1].clone(), cam_frames[i].clone()));
        }
    }

    println!("[*] Processing {} frame pairs in parallel...", frame_pairs.len());

    frame_pairs.par_iter().for_each(|(prev_info, curr_info)| {
        let prev_path = format!("{}/{}", images_folder, prev_info.image_file);
        let curr_path = format!("{}/{}", images_folder, curr_info.image_file);

        let mut prev_img = ImageGray::default();
        let mut curr_img = ImageGray::default();

        if !load_image_gray(&prev_path, &mut prev_img) {
            eprintln!("[!] Skipping pair due to load error: {}", prev_info.image_file);
            return;
        }

        if !load_image_gray(&curr_path, &mut curr_img) {
            eprintln!("[!] Skipping pair due to load error: {}", curr_info.image_file);
            return;
        }

        let mm = detect_motion(&prev_img, &curr_img, motion_threshold);

        if mm.width == 0 { return; }

        let cam_pos = curr_info.camera_position.clone();
        let cam_rot = rotation_matrix_yaw_pitch_roll(curr_info.yaw, curr_info.pitch, curr_info.roll);
        let fov_rad = deg2rad(curr_info.fov_degrees);
        let focal_len = (mm.width as f32 * 0.5) / (0.5 * fov_rad).tan();

        let pixel_indices: Vec<usize> = (0..mm.width * mm.height).map(|i| i as usize).collect();

        pixel_indices.par_iter().for_each(|&idx| {
            if !mm.changed[idx] {
                return;
            }
            let pix_val = mm.diff[idx];
            if pix_val < 1e-3 {
                return;
            }

            let u = (idx % mm.width as usize) as f32;
            let v = (idx / mm.width as usize) as f32;

            let mut ray_cam = Vec3 {
                x: u - 0.5 * mm.width as f32,
                y: -(v - 0.5 * mm.height as f32),
                z: -focal_len,
            };

            ray_cam = normalize(ray_cam);
            let mut ray_world = mat3_mul_vec3(&cam_rot, ray_cam);
            ray_world = normalize(ray_world);

            let steps = cast_ray_into_grid(cam_pos.clone(), ray_world, N as i32, VOXEL_SIZE, grid_center.clone());
            for rs in steps {
                // i am waiting for this implementation to understand fully the purpose
                // let dist = rs.distance;
                // let attenuation = 1.0 / (1.0 + alpha * dist);
                // let area_scale = 1.0 / (dist * dist + 1e-6);
                // let val = pix_val * attenuation * area_scale;
                let val = pix_val * 1.0;

                let grid_idx = rs.ix as usize * N * N + rs.iy as usize * N + rs.iz as usize;

                voxel_grid_ref[grid_idx].fetch_add(val, Ordering::Relaxed);
            }

            total_processed_pixels.fetch_add(1, Ordering::Relaxed);
        });
    });

    println!("[+] Frame processing completed in {:?}", start_processing.elapsed());

    let start_save = Instant::now();
    println!("[*] Saving voxel grid to file: {}", output_bin);
    let file = File::create(output_bin)?;
    let mut writer = BufWriter::new(file);

    writer.write_all(&(N as i32).to_le_bytes())?;
    writer.write_all(&VOXEL_SIZE.to_le_bytes())?;

    for val_atomic in voxel_grid.iter() {
        let val = val_atomic.load(Ordering::Relaxed);
        writer.write_all(&val.to_le_bytes())?;
    }

    writer.flush()?;
    println!("[+] Saved voxel grid in {:?}", start_save.elapsed());

    let final_processed_pixels = total_processed_pixels.load(Ordering::Relaxed);
    println!("[+] Total processed pixels: {}", final_processed_pixels);
    println!("[+] Total runtime: {:?}", start_total.elapsed());

    Ok(())
}
