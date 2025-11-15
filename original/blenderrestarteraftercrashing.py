#!/usr/bin/env python3
"""
Watch-dog that restarts Blender until every entry in metadata.json
is "rendered": true.  Now hardened against path-quoting issues and
third-party add-ons crashing on start-up.
"""

import json, os, subprocess, sys, tempfile, time
from pathlib import Path
import textwrap, json as _json  # for safe string literal

# ─────────────────────────── USER SETTINGS ────────────────────────────
BLENDER_EXE = r"C:\Program Files\Blender Foundation\Blender 3.6\blender.exe"
BLEND_FILE  = r"C:\Users\youruser\Desktop\birdge.blend"
JOB_SCRIPT  = r"C:\Users\youruser\Desktop\birdge.py"

OUTPUT_DIR  = r"C:\Users\Rhoady\Desktop\pet projects\relcams40"
META_PATH   = Path(OUTPUT_DIR) / "metadata.json"

CRASH_SLEEP = 5.0        # seconds between retries
# ───────────────────────────────────────────────────────────────────────

# ----------------------------------------------------------------------
def all_frames_done(meta_file: Path) -> bool:
    if not meta_file.exists():
        return False
    with meta_file.open("r") as fh:
        return all(m.get("rendered") for m in json.load(fh))

# ----------------------------------------------------------------------
def make_wrapper() -> Path:
    wrapper = Path(tempfile.gettempdir()) / "run_render_job_once.py"

    safe_job  = _json.dumps(JOB_SCRIPT)          # proper quoting
    safe_json = _json.dumps(str(META_PATH))      # ← new

    wrapper.write_text(textwrap.dedent(f"""
        import importlib.util, types, pathlib, sys, os

        # ---- import the render job script ----------------------------------
        job_path = pathlib.Path({safe_job})
        spec = importlib.util.spec_from_file_location("job", job_path)
        if spec and spec.loader:
            job = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(job)
        else:                                  # very old Python fallback
            job = types.ModuleType("job")
            exec(job_path.read_text(), job.__dict__)
        sys.modules["job"] = job

        # ---- pick fresh vs. resume ----------------------------------------
        meta_path = pathlib.Path({safe_json})
        job.NEW_PHOTOS = not meta_path.exists()    # True on first run only
        print(f"[WRAPPER] NEW_PHOTOS = {{job.NEW_PHOTOS}}")

        job.main()
    """))

    return wrapper

# ----------------------------------------------------------------------
def launch_blender(wrapper: Path) -> int:
    cmd = [
        BLENDER_EXE,
        "--factory-startup",          # <-- disables all add-ons
        "-b", BLEND_FILE,
        "--python", str(wrapper)
    ]
    return subprocess.run(cmd).returncode

# ----------------------------------------------------------------------
def main():
    print("▶  Blender render watchdog started.")
    wrapper = make_wrapper()

    while True:
        if all_frames_done(META_PATH):
            print("✓  All frames already rendered – nothing to do. Exiting.")
            return

        print("→ Launching Blender …")
        rc = launch_blender(wrapper)

        if rc == 0 and all_frames_done(META_PATH):
            print("✓  Job completed on this pass. Exiting.")
            return

        print(f"⚠  Blender exited (code {rc}). Will retry in {CRASH_SLEEP}s …")
        time.sleep(CRASH_SLEEP)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
