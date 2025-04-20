#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generates animated GIFs of zooming Julia sets.

Configuration is done by editing the PARAMETERS section in this script.
Includes Numba optimization for performance and smooth coloring algorithms.
"""

# No longer need argparse
import os
import time
import sys
from typing import Tuple, Optional, Dict, Any

import numpy as np
from PIL import Image
import imageio

# Attempt Numba import for performance
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True

    def conditional_jit(func):
        """Applies Numba JIT compilation with specific options."""
        return jit(nopython=True, parallel=True, cache=True)(func)

except ImportError:
    NUMBA_AVAILABLE = False
    prange = range
    def conditional_jit(func):
        """Dummy decorator when Numba is not available."""
        return func
    print("Warning: Numba not found. Calculations will be significantly slower.",
          file=sys.stderr)
    print("Install Numba for performance boost: pip install numba",
          file=sys.stderr)


# --- Constants (Used as defaults within generation function if needed) ---
DEFAULT_WIDTH: int = 600
DEFAULT_HEIGHT: int = 600
DEFAULT_MAX_ITER: int = 150
DEFAULT_ESCAPE_RADIUS: float = 2.0
DEFAULT_NUM_FRAMES: int = 250
DEFAULT_ZOOM_FACTOR: float = 0.96
DEFAULT_FRAME_DURATION: float = 0.04
DEFAULT_ZOOM_CENTER_X: float = 0.0
DEFAULT_ZOOM_CENTER_Y: float = 0.0
DEFAULT_X_MIN: float = -1.5
DEFAULT_X_MAX: float = 1.5
DEFAULT_Y_MIN: float = -1.5
DEFAULT_Y_MAX: float = 1.5
DEFAULT_HUE_CYCLE_FACTOR: float = 5.0
DEFAULT_SATURATION: float = 0.95
DEFAULT_VALUE: float = 0.90
DEFAULT_OUTPUT_DIR: str = "julia_gifs"


# --- Core Julia Set Calculation (Remains the same) ---
@conditional_jit
def compute_julia_iterations_numba(
    z_coords_real: np.ndarray,
    z_coords_imag: np.ndarray,
    c_param_real: float,
    c_param_imag: float,
    max_iter: int,
    escape_radius_sq: float
) -> Tuple[np.ndarray, np.ndarray]:
    """ Computes the number of iterations until points escape."""
    # Implementation is identical to previous versions
    height, width = z_coords_real.shape
    iterations = np.full((height, width), max_iter, dtype=np.int32)
    z_real = z_coords_real.copy()
    z_imag = z_coords_imag.copy()
    final_magnitudes = np.zeros((height, width), dtype=np.float64)

    for r in prange(height):
        for c in range(width):
            zr = z_real[r, c]
            zi = z_imag[r, c]
            for i in range(max_iter):
                zr_next = zr * zr - zi * zi + c_param_real
                zi_next = 2.0 * zr * zi + c_param_imag
                zr = zr_next
                zi = zi_next
                mag_sq = zr * zr + zi * zi
                if mag_sq > escape_radius_sq:
                    iterations[r, c] = i
                    final_magnitudes[r, c] = np.sqrt(mag_sq)
                    break
            else:
                iterations[r, c] = max_iter
                final_magnitudes[r, c] = np.sqrt(zr * zr + zi * zi)
    return iterations, final_magnitudes


# --- Coloring Functions (Remain the same) ---
# Optional Numba jit
# @conditional_jit
def hsv_to_rgb_vectorized(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """ Vectorized HSV to RGB conversion using NumPy operations."""
    # Implementation is identical to previous versions
    h = h % 1.0
    i = np.floor(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i.astype(np.int32) % 6
    rgb = np.zeros(h.shape + (3,), dtype=h.dtype)
    conditions = [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5]
    choices_r = [v, q, p, p, t, v]
    choices_g = [t, v, v, q, p, p]
    choices_b = [p, p, t, v, v, q]
    rgb[..., 0] = np.select(conditions, choices_r, default=v)
    rgb[..., 1] = np.select(conditions, choices_g, default=v)
    rgb[..., 2] = np.select(conditions, choices_b, default=v)
    return rgb

# Optional Numba jit
# @conditional_jit
def colorize_smooth_vectorized(
    iterations: np.ndarray,
    final_magnitudes: np.ndarray,
    max_iter: int,
    escape_radius: float,
    hue_cycle_factor: float = DEFAULT_HUE_CYCLE_FACTOR,
    saturation: float = DEFAULT_SATURATION,
    value: float = DEFAULT_VALUE
) -> np.ndarray:
    """ Generates an RGB image array using smooth coloring."""
    # Implementation is identical to previous versions
    height, width = iterations.shape
    img_array_rgb_float = np.zeros((height, width, 3), dtype=np.float64)
    outside_set = (iterations < max_iter)
    if not np.any(outside_set):
        return np.zeros((height, width, 3), dtype=np.uint8)

    iterations_outside = iterations[outside_set]
    magnitudes_outside = final_magnitudes[outside_set]
    safe_magnitudes = np.maximum(magnitudes_outside, escape_radius + 1e-9)
    log_zn = np.log(safe_magnitudes)
    log_escape = np.log(escape_radius)
    if abs(log_escape) < 1e-9: log_escape = 1e-9 # Avoid div by zero if radius is 1
    nu = np.log(log_zn / log_escape) / np.log(2.0)
    nu = np.nan_to_num(nu)
    smooth_val = iterations_outside + 1.0 - nu
    smooth_val = np.maximum(smooth_val, 0.0)
    hue = (smooth_val / max_iter * hue_cycle_factor) % 1.0
    sat_array = np.full_like(hue, saturation)
    val_array = np.full_like(hue, value)
    rgb_outside_float = hsv_to_rgb_vectorized(hue, sat_array, val_array)
    img_array_rgb_float[outside_set] = rgb_outside_float
    img_array_uint8 = (img_array_rgb_float * 255).astype(np.uint8)
    return img_array_uint8


# --- GIF Generation Function (Remains largely the same internally) ---
def generate_julia_zoom(
    c_value: complex,
    output_path: str,
    width: int = DEFAULT_WIDTH, # Use constants for defaults here now
    height: int = DEFAULT_HEIGHT,
    max_iter: int = DEFAULT_MAX_ITER,
    escape_radius: float = DEFAULT_ESCAPE_RADIUS,
    num_frames: int = DEFAULT_NUM_FRAMES,
    zoom_factor_per_frame: float = DEFAULT_ZOOM_FACTOR,
    frame_duration: float = DEFAULT_FRAME_DURATION,
    zoom_center_x: float = DEFAULT_ZOOM_CENTER_X,
    zoom_center_y: float = DEFAULT_ZOOM_CENTER_Y,
    x_min_start: float = DEFAULT_X_MIN,
    x_max_start: float = DEFAULT_X_MAX,
    y_min_start: float = DEFAULT_Y_MIN,
    y_max_start: float = DEFAULT_Y_MAX,
    color_options: Optional[Dict[str, Any]] = None
) -> None:
    """
    Generates and saves a zooming Julia set animation GIF.
    Uses parameters passed directly or falls back to constants defined above.
    """
    # --- Input Validation --- (Still useful)
    if width <= 0 or height <= 0: raise ValueError("Width/height must be positive.")
    if max_iter <= 0: raise ValueError("Max iterations must be positive.")
    if escape_radius <= 0: raise ValueError("Escape radius must be positive.")
    if num_frames <= 0: raise ValueError("Number of frames must be positive.")
    if not (0 < zoom_factor_per_frame < 1): raise ValueError("Zoom factor invalid.")
    if frame_duration <= 0: raise ValueError("Frame duration must be positive.")
    if x_max_start <= x_min_start or y_max_start <= y_min_start: raise ValueError("Max coords <= min coords.")

    print("-" * 50)
    print(f"Starting generation for C = {c_value}")
    print(f"Output target: {output_path}")
    # Print key parameters
    print(f"Resolution: {width}x{height}, Max Iter: {max_iter}, Frames: {num_frames}")
    print(f"Numba available: {NUMBA_AVAILABLE}")
    print("-" * 50)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            raise IOError(f"Error creating directory {output_dir}: {e}") from e

    # --- Initialization ---
    frames = []
    x_min, x_max = x_min_start, x_max_start
    y_min, y_max = y_min_start, y_max_start
    escape_radius_sq = escape_radius**2
    c_real, c_imag = c_value.real, c_value.imag

    # Prepare coloring args, using defaults if not provided
    coloring_args = {
        "max_iter": max_iter,
        "escape_radius": escape_radius,
        "hue_cycle_factor": DEFAULT_HUE_CYCLE_FACTOR,
        "saturation": DEFAULT_SATURATION,
        "value": DEFAULT_VALUE
    }
    if color_options: # Apply overrides if provided
        coloring_args.update(color_options)

    total_start_time = time.time()
    print(f"Generating {num_frames} frames...")

    # --- Main Loop (Identical to previous version) ---
    for i in range(num_frames):
        frame_start_time = time.time()
        real = np.linspace(x_min, x_max, width, dtype=np.float64)
        imag = np.linspace(y_min, y_max, height, dtype=np.float64)
        re_grid, im_grid = np.meshgrid(real, imag)
        iterations, final_magnitudes = compute_julia_iterations_numba(
            re_grid, im_grid, c_real, c_imag, max_iter, escape_radius_sq
        )
        img_array = colorize_smooth_vectorized(
            iterations, final_magnitudes, **coloring_args
        )
        img = Image.fromarray(img_array, 'RGB')
        frames.append(img)
        current_width = x_max - x_min
        current_height = y_max - y_min
        new_width = current_width * zoom_factor_per_frame
        new_height = current_height * zoom_factor_per_frame
        x_min = zoom_center_x - new_width / 2.0
        x_max = zoom_center_x + new_width / 2.0
        y_min = zoom_center_y - new_height / 2.0
        y_max = zoom_center_y + new_height / 2.0
        frame_end_time = time.time()
        if (i + 1) % max(1, num_frames // 10) == 0 or i == num_frames - 1:
            print(f"  Frame {i+1}/{num_frames} generated in "
                  f"{frame_end_time - frame_start_time:.3f} seconds.")

    # --- Save the GIF (Identical to previous version) ---
    print(f"\nSaving animation to {output_path}...")
    save_start_time = time.time()
    try:
        imageio.mimsave(output_path, frames, duration=frame_duration, loop=0,
                        subrectangles=True)
        save_end_time = time.time()
        print(f"GIF saved successfully in {save_end_time - save_start_time:.2f} seconds.")
    except Exception as e:
        raise IOError(f"Error saving GIF {output_path}: {e}") from e

    total_end_time = time.time()
    print(f"\nFinished generation for C = {c_value}")
    print(f"Total time for this GIF: {total_end_time - total_start_time:.2f} seconds.")
    print("-" * 50)


# --- Main Execution Block ---
if __name__ == "__main__":
    script_start_time = time.time()

    # --- Numba Pre-compilation ---
    if NUMBA_AVAILABLE:
        print("Checking Numba compilation status...")
        compile_start = time.time()
        try:
            # Dummy call to trigger compilation
            _ = compute_julia_iterations_numba(
                np.zeros((2,2), dtype=np.float64), np.zeros((2,2), dtype=np.float64),
                0.0, 0.0, 10, 4.0
            )
            compile_time = time.time() - compile_start
            if compile_time > 0.1: # Heuristic: check if it took time
                print(f"Numba compilation finished in {compile_time:.2f} seconds.")
            else:
                print("Numba function likely loaded from cache.")
        except Exception as e:
            print(f"Warning: Numba compilation failed. Error: {e}", file=sys.stderr)

    # --- Define GIF Generation Parameters Here ---
    # Define one or more dictionaries, each representing a GIF to generate.
    # The keys should match the arguments of `generate_julia_zoom`.
    # If a key is missing, the default value from the constants/function signature will be used.

    gif_tasks = [
        {
            "c_value": complex(-0.7269, 0.1889),
            "output_path": os.path.join(DEFAULT_OUTPUT_DIR, "julia_c1_script_config.gif"),
            # Example: Override default frames
            "num_frames": 150,
        },
        {
            "c_value": complex(-0.8, 0.156),
            "output_path": os.path.join(DEFAULT_OUTPUT_DIR, "julia_c2_script_config.gif"),
            # Example: Override resolution and coloring
            "width": 800,
            "height": 800,
            "color_options": {
                "hue_cycle_factor": 8.0,
                "saturation": 0.85
            }
        },
        {
             # Example: Using mostly defaults defined above
             "c_value": complex(0.285, 0.01),
             "output_path": os.path.join(DEFAULT_OUTPUT_DIR, "julia_c3_defaults.gif"),
        }
    ]

    # --- Run the Generation Tasks ---
    print(f"\nFound {len(gif_tasks)} GIF generation tasks defined in script.")
    for i, params in enumerate(gif_tasks):
        print(f"\n--- Starting Task {i+1} of {len(gif_tasks)} ---")
        try:
            # Unpack the dictionary as keyword arguments
            generate_julia_zoom(**params)
        except (ValueError, IOError) as e:
            print(f"\nError generating GIF for C={params.get('c_value', 'N/A')}: {e}",
                  file=sys.stderr)
            # Continue to the next task
        except Exception as e:
            # Catch unexpected errors
            print(f"\nAn unexpected error occurred during task {i+1}: {e}",
                   file=sys.stderr)
            import traceback
            traceback.print_exc()
            # Decide whether to continue or stop
            # continue # or break or sys.exit(1)

    script_end_time = time.time()
    print("\nAll defined generation tasks complete.")
    print(f"Total script execution time: {script_end_time - script_start_time:.2f} seconds.")