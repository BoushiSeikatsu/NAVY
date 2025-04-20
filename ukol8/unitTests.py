
**3. Basic Test Suite (`test_julia_generator.py`)**

```python
import unittest
import numpy as np
import os
import tempfile
import shutil

# Import functions to be tested from the main script
# Assuming the main script is named 'julia_generator.py'
try:
    from julia_generator import (
        hsv_to_rgb_vectorized,
        compute_julia_iterations_numba, # Will use Numba if available
        generate_julia_zoom,
        parse_complex
    )
    # We need the constants for default args if testing generate function directly
    from julia_generator import (
         DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_MAX_ITER,
         DEFAULT_ESCAPE_RADIUS, DEFAULT_NUM_FRAMES, DEFAULT_ZOOM_FACTOR,
         DEFAULT_FRAME_DURATION, DEFAULT_ZOOM_CENTER_X, DEFAULT_ZOOM_CENTER_Y,
         DEFAULT_X_MIN, DEFAULT_X_MAX, DEFAULT_Y_MIN, DEFAULT_Y_MAX
    )
except ImportError:
    raise ImportError("Could not import from julia_generator.py. "
                      "Ensure it's in the same directory or Python path.")


# Tolerance for floating point comparisons
FLOAT_TOLERANCE = 1e-6

class TestJuliaGenerator(unittest.TestCase):

    def test_hsv_to_rgb_conversion(self):
        """Test the vectorized HSV to RGB conversion for known values."""
        # Test Black (H=0, S=0, V=0) -> (0, 0, 0)
        h = np.array([0.0])
        s = np.array([0.0])
        v = np.array([0.0])
        rgb = hsv_to_rgb_vectorized(h, s, v)
        np.testing.assert_allclose(rgb[0], [0.0, 0.0, 0.0], atol=FLOAT_TOLERANCE)

        # Test White (H=any, S=0, V=1) -> (1, 1, 1)
        h = np.array([0.5])
        s = np.array([0.0])
        v = np.array([1.0])
        rgb = hsv_to_rgb_vectorized(h, s, v)
        np.testing.assert_allclose(rgb[0], [1.0, 1.0, 1.0], atol=FLOAT_TOLERANCE)

        # Test Red (H=0, S=1, V=1) -> (1, 0, 0)
        h = np.array([0.0 / 6.0])
        s = np.array([1.0])
        v = np.array([1.0])
        rgb = hsv_to_rgb_vectorized(h, s, v)
        np.testing.assert_allclose(rgb[0], [1.0, 0.0, 0.0], atol=FLOAT_TOLERANCE)

        # Test Green (H=120/360=1/3, S=1, V=1) -> (0, 1, 0)
        h = np.array([1.0 / 3.0])
        s = np.array([1.0])
        v = np.array([1.0])
        rgb = hsv_to_rgb_vectorized(h, s, v)
        np.testing.assert_allclose(rgb[0], [0.0, 1.0, 0.0], atol=FLOAT_TOLERANCE)

        # Test Blue (H=240/360=2/3, S=1, V=1) -> (0, 0, 1)
        h = np.array([2.0 / 3.0])
        s = np.array([1.0])
        v = np.array([1.0])
        rgb = hsv_to_rgb_vectorized(h, s, v)
        np.testing.assert_allclose(rgb[0], [0.0, 0.0, 1.0], atol=FLOAT_TOLERANCE)

        # Test multiple values at once
        h = np.array([0.0, 1.0/3.0, 2.0/3.0])
        s = np.array([1.0, 1.0, 0.0])
        v = np.array([1.0, 1.0, 0.5])
        expected_rgb = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.5]])
        rgb = hsv_to_rgb_vectorized(h, s, v)
        np.testing.assert_allclose(rgb, expected_rgb, atol=FLOAT_TOLERANCE)

    def test_core_julia_iterations_simple(self):
        """Test basic behavior of the core iteration function."""
        max_iter = 10
        escape_r_sq = 4.0 # (radius 2)^2

        # Test c = 0, z = 0 (should stay at origin, never escape)
        z_real = np.array([[0.0]])
        z_imag = np.array([[0.0]])
        iters, mags = compute_julia_iterations_numba(
            z_real, z_imag, 0.0, 0.0, max_iter, escape_r_sq
        )
        self.assertEqual(iters[0, 0], max_iter)
        self.assertAlmostEqual(mags[0, 0], 0.0, delta=FLOAT_TOLERANCE)

        # Test c = 0, z = 3 (should escape immediately, iter 0)
        z_real = np.array([[3.0]])
        z_imag = np.array([[0.0]])
        iters, mags = compute_julia_iterations_numba(
            z_real, z_imag, 0.0, 0.0, max_iter, escape_r_sq
        )
        # Note: Escape check happens *after* first iteration z=z^2+c is calculated
        # z0=3 -> z1=9 -> escapes. Check happens for i=0.
        self.assertEqual(iters[0, 0], 0)
        # Magnitude should be |z1| = 9.0
        self.assertAlmostEqual(mags[0, 0], 9.0, delta=FLOAT_TOLERANCE)

        # Test c = -1, z = 0 ( z0=0, z1=-1, z2=0, z3=-1 ... stays bounded)
        z_real = np.array([[0.0]])
        z_imag = np.array([[0.0]])
        iters, mags = compute_julia_iterations_numba(
            z_real, z_imag, -1.0, 0.0, max_iter, escape_r_sq
        )
        self.assertEqual(iters[0, 0], max_iter)
        # Final mag alternates 0 or 1 depending on max_iter even/odd
        self.assertTrue(abs(mags[0, 0] - 0.0) < FLOAT_TOLERANCE or
                        abs(mags[0, 0] - 1.0) < FLOAT_TOLERANCE)

    def test_parse_complex_valid(self):
        """Test complex number parsing for valid inputs."""
        self.assertEqual(parse_complex("1+2j"), 1+2j)
        self.assertEqual(parse_complex("-0.5-0.1j"), -0.5-0.1j)
        self.assertEqual(parse_complex("3j"), 3j)
        self.assertEqual(parse_complex("-4"), -4+0j)
        self.assertEqual(parse_complex(" -0.8 + 0.156j "), -0.8+0.156j) # Handles spaces

    def test_parse_complex_invalid(self):
        """Test complex number parsing raises error for invalid inputs."""
        from argparse import ArgumentTypeError
        with self.assertRaises(ArgumentTypeError):
            parse_complex("1+j2")
        with self.assertRaises(ArgumentTypeError):
            parse_complex("abc")
        with self.assertRaises(ArgumentTypeError):
            parse_complex("1+2i") # Requires 'j'

    # Optional: Add a test for generate_julia_zoom, but this is more complex
    # as it involves file I/O. Using a temporary directory is recommended.
    def test_generate_gif_runs(self):
        """ Test if the main generation function runs without crashing
            and creates an output file (basic check)."""
        # Create a temporary directory for output
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "test_julia.gif")
        try:
            generate_julia_zoom(
                c_value=complex(-0.8, 0.156),
                output_path=output_path,
                width=20, # Use small dimensions for speed
                height=20,
                max_iter=10,
                escape_radius=DEFAULT_ESCAPE_RADIUS,
                num_frames=3, # Generate only a few frames
                zoom_factor_per_frame=DEFAULT_ZOOM_FACTOR,
                frame_duration=DEFAULT_FRAME_DURATION,
                zoom_center_x=DEFAULT_ZOOM_CENTER_X,
                zoom_center_y=DEFAULT_ZOOM_CENTER_Y,
                x_min_start=DEFAULT_X_MIN,
                x_max_start=DEFAULT_X_MAX,
                y_min_start=DEFAULT_Y_MIN,
                y_max_start=DEFAULT_Y_MAX
            )
            # Check if the file was created
            self.assertTrue(os.path.exists(output_path))
            # Check if the file has non-zero size (basic check for content)
            self.assertGreater(os.path.getsize(output_path), 0)
        finally:
            # Clean up the temporary directory and its contents
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()