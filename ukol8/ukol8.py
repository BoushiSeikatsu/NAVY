"""
Generates animated GIFs of zooming Julia sets.

Configuration is done by editing the PARAMETERS section in this script.
Includes Numba optimization for performance and smooth coloring algorithms.
"""

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
    # Define a conditional decorator for Numba JIT compilation.
    # Options: nopython=True (strict, faster), parallel=True (uses multiple cores), cache=True (saves compiled version).
    def conditional_jit(func):
        """Applies Numba JIT compilation with specific options."""
        return jit(nopython=True, parallel=True, cache=True)(func)

except ImportError:
    NUMBA_AVAILABLE = False
    # Provide a fallback if Numba is not installed. `prange` becomes standard `range`.
    prange = range
    # Define a dummy decorator that does nothing if Numba is unavailable.
    def conditional_jit(func):
        """Dummy decorator when Numba is not available."""
        return func
    print("Warning: Numba not found. Calculations will be significantly slower.",
          file=sys.stderr)
    print("Install Numba for performance boost: pip install numba",
          file=sys.stderr)


# --- Constants (Used as defaults within generation function if needed) ---
# Define default configuration parameters for the generation process.
DEFAULT_WIDTH: int = 600
DEFAULT_HEIGHT: int = 600
DEFAULT_MAX_ITER: int = 150
DEFAULT_ESCAPE_RADIUS: float = 2.0
DEFAULT_NUM_FRAMES: int = 250
DEFAULT_ZOOM_FACTOR: float = 0.96 # Must be < 1 for zooming in
DEFAULT_FRAME_DURATION: float = 0.04 # In seconds
DEFAULT_ZOOM_CENTER_X: float = 0.0
DEFAULT_ZOOM_CENTER_Y: float = 0.0
DEFAULT_X_MIN: float = -1.5
DEFAULT_X_MAX: float = 1.5
DEFAULT_Y_MIN: float = -1.5
DEFAULT_Y_MAX: float = 1.5
DEFAULT_HUE_CYCLE_FACTOR: float = 5.0 # Controls color frequency
DEFAULT_SATURATION: float = 0.95
DEFAULT_VALUE: float = 0.90 # Brightness component
DEFAULT_OUTPUT_DIR: str = "julia_gifs"


# --- Core Julia Set Calculation ---
# Apply Numba JIT for parallel execution and caching to accelerate this critical function.
@conditional_jit
def compute_julia_iterations_numba(
    z_coords_real: np.ndarray,
    z_coords_imag: np.ndarray,
    c_param_real: float,
    c_param_imag: float,
    max_iter: int,
    escape_radius_sq: float
) -> Tuple[np.ndarray, np.ndarray]:
    """ Computes the number of iterations until points escape for the Julia set formula z = z^2 + c."""
    height, width = z_coords_real.shape
    # Initialize iteration count to max_iter (assuming point is in the set initially).
    iterations = np.full((height, width), max_iter, dtype=np.int32)
    z_real = z_coords_real.copy() # Start z with the coordinate value
    z_imag = z_coords_imag.copy()
    # Store final magnitudes for smooth coloring calculation.
    final_magnitudes = np.zeros((height, width), dtype=np.float64)

    # Using Numba's prange for potential parallel execution over rows.
    for r in prange(height):
        for c in range(width):
            zr = z_real[r, c]
            zi = z_imag[r, c]
            # Iterate the Julia formula up to max_iter times.
            for i in range(max_iter):
                # Store previous zr to avoid using updated zr in zi calculation.
                # Core Julia set iteration formula: z_next = z^2 + c
                # Real part: zr*zr - zi*zi + c_real
                zr_next = zr * zr - zi * zi + c_param_real
                # Imaginary part: 2*zr*zi + c_imag
                zi_next = 2.0 * zr * zi + c_param_imag
                zr = zr_next
                zi = zi_next
                # Calculate squared magnitude (more efficient than sqrt).
                mag_sq = zr * zr + zi * zi
                # Check if the point has escaped the escape radius.
                if mag_sq > escape_radius_sq:
                    iterations[r, c] = i # Record the iteration count at escape.
                    final_magnitudes[r, c] = np.sqrt(mag_sq) # Store the magnitude.
                    break # Stop iterating for this point.
            else:
                # If the loop completes without breaking, the point is considered part of the set.
                iterations[r, c] = max_iter
                # Store final magnitude even for points inside the set.
                final_magnitudes[r, c] = np.sqrt(zr * zr + zi * zi)
    return iterations, final_magnitudes


# --- Coloring Functions ---
# Note: Numba JIT can be applied here too, but gains might be less significant than the main compute loop.
# @conditional_jit
def hsv_to_rgb_vectorized(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """ Convert Hue, Saturation, Value (HSV) arrays to RGB using vectorized NumPy operations."""
    h = h % 1.0 # Ensure hue is within [0, 1)
    i = np.floor(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i.astype(np.int32) % 6
    rgb = np.zeros(h.shape + (3,), dtype=h.dtype) # Prepare output RGB array

    # Select RGB components based on the hue sector using NumPy's `select`.
    conditions = [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5]
    choices_r = [v, q, p, p, t, v]
    choices_g = [t, v, v, q, p, p]
    choices_b = [p, p, t, v, v, q]
    rgb[..., 0] = np.select(conditions, choices_r, default=v)
    rgb[..., 1] = np.select(conditions, choices_g, default=v)
    rgb[..., 2] = np.select(conditions, choices_b, default=v)
    return rgb

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
    """ Generates an RGB image array using a smooth coloring algorithm based on iteration count and final magnitude."""
    height, width = iterations.shape
    img_array_rgb_float = np.zeros((height, width, 3), dtype=np.float64)
    # Create a boolean mask for points that escaped (iterations < max_iter).
    outside_set = (iterations < max_iter)

    # Avoid calculations if all points are inside the set (all black).
    if not np.any(outside_set):
        return np.zeros((height, width, 3), dtype=np.uint8)

    # Select data only for points outside the set.
    iterations_outside = iterations[outside_set]
    magnitudes_outside = final_magnitudes[outside_set]

    # Ensure magnitudes are slightly larger than escape_radius to avoid log(0) or log(<1) issues.
    safe_magnitudes = np.maximum(magnitudes_outside, escape_radius + 1e-9)

    # Calculate the smooth iteration value 'nu' based on final magnitude and escape radius.
    # This value interpolates between integer iteration counts.
    log_zn = np.log(safe_magnitudes)
    log_escape = np.log(escape_radius)
    if abs(log_escape) < 1e-9: log_escape = 1e-9 # Avoid division by zero if radius is 1.0
    nu = np.log(log_zn / log_escape) / np.log(2.0)
    # Handle potential NaN values resulting from calculations.
    nu = np.nan_to_num(nu)

    # Combine integer iterations with the fractional part 'nu'.
    smooth_val = iterations_outside + 1.0 - nu
    smooth_val = np.maximum(smooth_val, 0.0) # Ensure non-negative value.

    # Normalize the smoothed value and apply hue cycle factor to determine the hue.
    hue = (smooth_val / max_iter * hue_cycle_factor) % 1.0
    # Create arrays for saturation and value for vectorized HSV conversion.
    sat_array = np.full_like(hue, saturation)
    val_array = np.full_like(hue, value)

    # Convert HSV colors to RGB for the escaped points.
    rgb_outside_float = hsv_to_rgb_vectorized(hue, sat_array, val_array)
    # Place the calculated RGB values into the corresponding positions in the full image array.
    img_array_rgb_float[outside_set] = rgb_outside_float

    # Convert final float RGB values (range [0,1]) to 8-bit integers (range [0,255]) for image display.
    img_array_uint8 = (img_array_rgb_float * 255).astype(np.uint8)
    return img_array_uint8


# --- GIF Generation Function ---
def generate_julia_zoom(
    c_value: complex,
    output_path: str,
    width: int = DEFAULT_WIDTH,
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
    Orchestrates the frame generation loop, calling computation and coloring functions.
    """
    # --- Input Validation ---
    # Perform basic validation of input parameters to prevent errors during generation.
    if width <= 0 or height <= 0: raise ValueError("Width/height must be positive.")
    if max_iter <= 0: raise ValueError("Max iterations must be positive.")
    if escape_radius <= 0: raise ValueError("Escape radius must be positive.")
    if num_frames <= 0: raise ValueError("Number of frames must be positive.")
    if not (0 < zoom_factor_per_frame < 1): raise ValueError("Zoom factor invalid (must be 0 < factor < 1).")
    if frame_duration <= 0: raise ValueError("Frame duration must be positive.")
    if x_max_start <= x_min_start or y_max_start <= y_min_start: raise ValueError("Max coords must be greater than min coords.")

    print("-" * 50)
    print(f"Starting generation for C = {c_value}")
    print(f"Output target: {output_path}")
    print(f"Resolution: {width}x{height}, Max Iter: {max_iter}, Frames: {num_frames}")
    print(f"Numba available: {NUMBA_AVAILABLE}")
    print("-" * 50)

    # Ensure the specified output directory exists before saving frames/GIF.
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            raise IOError(f"Error creating directory {output_dir}: {e}") from e

    # --- Initialization ---
    frames = [] # List to hold each generated frame (as PIL Image).
    # Initialize coordinate bounds for the first frame.
    x_min, x_max = x_min_start, x_max_start
    y_min, y_max = y_min_start, y_max_start
    # Pre-calculate squared escape radius for efficiency in the compute function.
    escape_radius_sq = escape_radius**2
    c_real, c_imag = c_value.real, c_value.imag

    # Prepare arguments dictionary for the coloring function, using defaults and applying overrides.
    coloring_args = {
        "max_iter": max_iter,
        "escape_radius": escape_radius,
        "hue_cycle_factor": DEFAULT_HUE_CYCLE_FACTOR,
        "saturation": DEFAULT_SATURATION,
        "value": DEFAULT_VALUE
    }
    if color_options: # Update with user-provided color options if available.
        coloring_args.update(color_options)

    total_start_time = time.time()
    print(f"Generating {num_frames} frames...")

    # --- Main Loop ---
    # Iterate through each frame of the animation.
    for i in range(num_frames):
        frame_start_time = time.time()

        # Generate complex plane coordinates corresponding to the pixel grid for the current zoom level.
        # `linspace` creates evenly spaced points within the current coordinate bounds.
        real = np.linspace(x_min, x_max, width, dtype=np.float64)
        imag = np.linspace(y_min, y_max, height, dtype=np.float64)
        # `meshgrid` creates 2D arrays representing the real and imaginary parts for each pixel.
        re_grid, im_grid = np.meshgrid(real, imag)

        # Compute iterations using the Numba-optimized function. This is the most compute-intensive part.
        iterations, final_magnitudes = compute_julia_iterations_numba(
            re_grid, im_grid, c_real, c_imag, max_iter, escape_radius_sq
        )

        # Colorize the iteration data using the smooth algorithm.
        img_array = colorize_smooth_vectorized(
            iterations, final_magnitudes, **coloring_args # Pass coloring arguments unpacked.
        )

        # Convert the NumPy array to a PIL Image object.
        img = Image.fromarray(img_array, 'RGB')
        # Append the generated PIL Image frame to the list.
        frames.append(img)

        # Update coordinate bounds for the next frame's zoom level.
        # Calculate the current width and height of the complex plane view.
        current_width = x_max - x_min
        current_height = y_max - y_min
        # Calculate the new width and height based on the zoom factor.
        new_width = current_width * zoom_factor_per_frame
        new_height = current_height * zoom_factor_per_frame
        # Recalculate the min/max coordinates centered around the zoom point.
        x_min = zoom_center_x - new_width / 2.0
        x_max = zoom_center_x + new_width / 2.0
        y_min = zoom_center_y - new_height / 2.0
        y_max = zoom_center_y + new_height / 2.0

        frame_end_time = time.time()
        # Provide progress updates periodically.
        if (i + 1) % max(1, num_frames // 10) == 0 or i == num_frames - 1:
            #print(f"  Frame {i+1}/{num_frames} generated in {frame_end_time - frame_start_time:.3f} seconds.")
            print(f"  Frame {i+1}/{num_frames} generated")

    # --- Save the GIF ---
    print(f"\nSaving animation to {output_path}...")
    save_start_time = time.time()
    try:
        # Save the collected frames as an animated GIF using imageio.
        # `duration` is per frame, `loop=0` means infinite loop.
        # `subrectangles=True` is an optimization for GIF saving.
        imageio.mimsave(output_path, frames, duration=frame_duration, loop=0,
                        subrectangles=True)
        save_end_time = time.time()
        print(f"GIF saved successfully in {save_end_time - save_start_time:.2f} seconds.")
    except Exception as e:
        # Handle potential errors during file saving.
        raise IOError(f"Error saving GIF {output_path}: {e}") from e

    total_end_time = time.time()
    print(f"\nFinished generation for C = {c_value}")
    print(f"Total time for this GIF: {total_end_time - total_start_time:.2f} seconds.")
    print("-" * 50)


# --- Main Execution Block ---
# This code runs only when the script is executed directly.
if __name__ == "__main__":
    script_start_time = time.time()

    # --- Numba Pre-compilation ---
    # Attempt to pre-compile the Numba function for potentially faster execution during the main loop.
    # This avoids compilation overhead on the first frame generation.
    if NUMBA_AVAILABLE:
        print("Checking Numba compilation status...")
        compile_start = time.time()
        try:
            # Perform a small, dummy call to trigger Numba's JIT compilation.
            _ = compute_julia_iterations_numba(
                np.zeros((2,2), dtype=np.float64), np.zeros((2,2), dtype=np.float64),
                0.0, 0.0, 10, 4.0
            )
            compile_time = time.time() - compile_start
            # Check if compilation actually occurred (took some time) or if it loaded from cache.
            if compile_time > 0.1: # Heuristic threshold
                print(f"Numba compilation finished in {compile_time:.2f} seconds.")
            else:
                print("Numba function likely loaded from cache.")
        except Exception as e:
            # Warn if compilation fails, but allow the script to continue (using slower, non-Numba code).
            print(f"Warning: Numba compilation failed. Error: {e}", file=sys.stderr)

    # --- Define GIF Generation Parameters Here ---
    # Define a list of dictionaries, each specifying parameters for a single GIF generation task.
    # Keys in the dictionary correspond to arguments of the `generate_julia_zoom` function.
    # If a key is missing, the default value defined earlier will be used.
    gif_tasks = [
        {
            "c_value": complex(-0.7269, 0.1889), # The 'c' constant defining this specific Julia set.
            "output_path": os.path.join(DEFAULT_OUTPUT_DIR, "julia_c1_script_config.gif"),
            # Example: Override the default number of frames.
            "num_frames": 450,
        },
        {
            "c_value": complex(-0.8, 0.156),
            "output_path": os.path.join(DEFAULT_OUTPUT_DIR, "julia_c2_script_config.gif"),
            # Example: Override resolution and some coloring options.
            "width": 800,
            "height": 800,
            "zoom_center_x": -0.17,
            "zoom_center_y": 0.15,
            "num_frames": 200,
            "color_options": { # Pass a dictionary for color parameters.
                "hue_cycle_factor": 8.0,
                "saturation": 0.85
            }
        },
        {
             # Example: Using mostly default parameters defined as constants above.
             "c_value": complex(0.285, 0.01), # Another 'c' value.
             "output_path": os.path.join(DEFAULT_OUTPUT_DIR, "julia_c3_defaults.gif"),
        }
        # Add more dictionaries here to generate more GIFs with different parameters.
    ]

    # --- Run the Generation Tasks ---
    print(f"\nFound {len(gif_tasks)} GIF generation tasks defined in script.")
    # Iterate through the defined tasks and execute generation for each set of parameters.
    for i, params in enumerate(gif_tasks):
        print(f"\n--- Starting Task {i+1} of {len(gif_tasks)} ---")
        try:
            # Unpack the task parameter dictionary as keyword arguments for the generation function.
            generate_julia_zoom(**params)
        except (ValueError, IOError) as e:
            # Catch expected errors (e.g., bad parameters, file issues) and report them.
            print(f"\nError generating GIF for C={params.get('c_value', 'N/A')}: {e}",
                  file=sys.stderr)
            # Continue to the next task even if one fails.
        except Exception as e:
            # Catch any unexpected errors during generation.
            print(f"\nAn unexpected error occurred during task {i+1}: {e}",
                   file=sys.stderr)
            import traceback
            traceback.print_exc() # Print stack trace for debugging.
            # Consider whether to continue or halt execution on unexpected errors.
            # continue # or break or sys.exit(1)

    script_end_time = time.time()
    print("\nAll defined generation tasks complete.")
    print(f"Total script execution time: {script_end_time - script_start_time:.2f} seconds.")