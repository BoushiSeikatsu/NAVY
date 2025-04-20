import tkinter as tk
from tkinter import ttk, colorchooser, messagebox
import random
import math

# --- Core Fractal Generation Logic ---

def generate_fractal_line(p1, p2, iterations, offset_size, roughness=0.5):
    """
    Recursively generates fractal points between p1 and p2 using midpoint displacement.

    Args:
        p1 (tuple): Start point (x1, y1).
        p2 (tuple): End point (x2, y2).
        iterations (int): Remaining number of iterations.
        offset_size (float): Current maximum displacement magnitude.
        roughness (float): Factor to reduce offset_size in each recursive step (0.0 to 1.0).
                           Lower values make it smoother faster, higher values retain roughness longer.
                           0.5 corresponds to halving the offset each time.

    Returns:
        list: A list of generated points (tuples) between p1 and p2, excluding p1 and p2.
    """
    if iterations <= 0:
        return [] # Base case: no more iterations, return no points

    # 1. Find Midpoint
    x1, y1 = p1
    x2, y2 = p2
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2

    # 2. Calculate Displacement (Vertical perturbation as per instructions)
    # The displacement amount is random within the current offset range.
    # The instructions imply a 50/50 chance of up/down, achieved by random.uniform range.
    displacement = random.uniform(-offset_size, offset_size)

    # 3. Apply Displacement
    displaced_mid_y = mid_y + displacement
    displaced_mid_point = (mid_x, displaced_mid_y)

    # 4. Calculate offset for the next iteration level
    # Reduce the offset range for the next level of detail.
    next_offset_size = offset_size * roughness

    # 5. Recursive calls for the two new segments
    # Generate points for the left segment (p1 to displaced_mid_point)
    points_left = generate_fractal_line(p1, displaced_mid_point, iterations - 1, next_offset_size, roughness)
    # Generate points for the right segment (displaced_mid_point to p2)
    points_right = generate_fractal_line(displaced_mid_point, p2, iterations - 1, next_offset_size, roughness)

    # Combine the results: points from left recursion + the new midpoint + points from right recursion
    return points_left + [displaced_mid_point] + points_right

# --- GUI Application Class ---

class FractalTerrainApp:
    """
    Main application class for the Fractal Terrain Generator GUI.
    """
    def __init__(self, master):
        """
        Initializes the application window and UI elements.
        """
        self.master = master
        master.title("Fractal Terrain 2D Generator")
        # Make window slightly resizable (optional)
        # master.geometry("1100x700") # Start size
        # master.minsize(950, 600)

        # --- Configuration ---
        self.canvas_width = 900
        self.canvas_height = 600
        self.control_width = 200
        self.default_color = "#000000"
        self.selected_color = tk.StringVar(value=self.default_color)
        self.roughness = tk.DoubleVar(value=0.6) # Added roughness control

        # --- Main Layout Frames ---
        self.main_frame = ttk.Frame(master, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Canvas on the left
        self.canvas_frame = ttk.Frame(self.main_frame)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Controls on the right
        self.control_frame = ttk.Frame(self.main_frame, width=self.control_width, padding="10")
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, expand=False)
        self.control_frame.pack_propagate(False) # Prevent controls from shrinking the frame

        # --- Canvas Setup ---
        self.canvas = tk.Canvas(self.canvas_frame, bg="white", width=self.canvas_width, height=self.canvas_height,
                                scrollregion=(0, 0, self.canvas_width, self.canvas_height)) # Set scroll region for potential future use

        # Optional: Add scrollbars if canvas content might exceed view
        # hbar = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        # vbar = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        # self.canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)

        # hbar.pack(side=tk.BOTTOM, fill=tk.X)
        # vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # --- Control Widgets ---
        self.setup_controls()

    def setup_controls(self):
        """Creates and places the control widgets in the control frame."""
        frame = self.control_frame
        row_index = 0

        # --- Input Fields ---
        ttk.Label(frame, text="Start X (float):").grid(row=row_index, column=0, sticky="w", pady=2)
        self.start_x_var = tk.StringVar(value="0")
        ttk.Entry(frame, textvariable=self.start_x_var).grid(row=row_index, column=1, sticky="ew", pady=2)
        row_index += 1

        ttk.Label(frame, text="Start Y (float):").grid(row=row_index, column=0, sticky="w", pady=2)
        self.start_y_var = tk.StringVar(value=str(self.canvas_height / 2)) # Default to middle
        ttk.Entry(frame, textvariable=self.start_y_var).grid(row=row_index, column=1, sticky="ew", pady=2)
        row_index += 1

        ttk.Label(frame, text="End X (float):").grid(row=row_index, column=0, sticky="w", pady=2)
        self.end_x_var = tk.StringVar(value=str(self.canvas_width)) # Default to canvas width
        ttk.Entry(frame, textvariable=self.end_x_var).grid(row=row_index, column=1, sticky="ew", pady=2)
        row_index += 1

        ttk.Label(frame, text="End Y (float):").grid(row=row_index, column=0, sticky="w", pady=2)
        self.end_y_var = tk.StringVar(value=str(self.canvas_height / 2)) # Default to middle
        ttk.Entry(frame, textvariable=self.end_y_var).grid(row=row_index, column=1, sticky="ew", pady=2)
        row_index += 1

        ttk.Separator(frame, orient=tk.HORIZONTAL).grid(row=row_index, column=0, columnspan=2, sticky="ew", pady=5)
        row_index += 1

        ttk.Label(frame, text="Iterations (int):").grid(row=row_index, column=0, sticky="w", pady=2)
        self.iterations_var = tk.StringVar(value="8") # Increased default for more detail
        ttk.Entry(frame, textvariable=self.iterations_var).grid(row=row_index, column=1, sticky="ew", pady=2)
        row_index += 1

        ttk.Label(frame, text="Offset Size (float):").grid(row=row_index, column=0, sticky="w", pady=2)
        self.offset_var = tk.StringVar(value="50") # Initial vertical range scale
        ttk.Entry(frame, textvariable=self.offset_var).grid(row=row_index, column=1, sticky="ew", pady=2)
        row_index += 1

        # Enhancement: Roughness control
        ttk.Label(frame, text="Roughness (0-1):").grid(row=row_index, column=0, sticky="w", pady=2)
        self.roughness_scale = ttk.Scale(frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL, variable=self.roughness)
        self.roughness_scale.grid(row=row_index, column=1, sticky="ew", pady=2)
        # Optional: Display roughness value
        # self.roughness_label = ttk.Label(frame, textvariable=self.roughness)
        # self.roughness_label.grid(row=row_index+1, column=1, sticky="e", pady=0)
        row_index += 2 # Increment by 2 if label added

        ttk.Separator(frame, orient=tk.HORIZONTAL).grid(row=row_index, column=0, columnspan=2, sticky="ew", pady=5)
        row_index += 1

        # --- Color Selection ---
        ttk.Label(frame, text="Selected Color:").grid(row=row_index, column=0, sticky="w", pady=2)
        self.color_display = tk.Label(frame, textvariable=self.selected_color, relief="sunken", width=10,
                                      bg=self.selected_color.get()) # Show color visually
        self.color_display.grid(row=row_index, column=1, sticky="ew", pady=2)
        # Update display background when color variable changes
        self.selected_color.trace_add("write", self.update_color_display)
        row_index += 1

        self.pick_color_button = ttk.Button(frame, text="Pick Color", command=self.pick_color_command)
        self.pick_color_button.grid(row=row_index, column=0, columnspan=2, sticky="ew", pady=5)
        row_index += 1

        ttk.Separator(frame, orient=tk.HORIZONTAL).grid(row=row_index, column=0, columnspan=2, sticky="ew", pady=5)
        row_index += 1

        # --- Action Buttons ---
        self.draw_button = ttk.Button(frame, text="Draw Terrain Layer", command=self.draw_command)
        self.draw_button.grid(row=row_index, column=0, columnspan=2, sticky="ew", pady=5)
        row_index += 1

        self.clear_button = ttk.Button(frame, text="Clear Canvas", command=self.clear_command)
        self.clear_button.grid(row=row_index, column=0, columnspan=2, sticky="ew", pady=5)
        row_index += 1

        # Configure column weights for resizing within control frame
        frame.columnconfigure(1, weight=1)

    def update_color_display(self, *args):
        """Updates the background color of the color display label."""
        try:
            self.color_display.config(bg=self.selected_color.get())
        except tk.TclError: # Handle case where color name is invalid momentarily
            self.color_display.config(bg=self.default_color) # Revert to default

    def pick_color_command(self):
        """Opens the color chooser dialog and updates the selected color."""
        color_code = colorchooser.askcolor(title="Choose Terrain Color", initialcolor=self.selected_color.get())
        if color_code and color_code[1]: # Check if a color was chosen (returns tuple (rgb, hex))
            self.selected_color.set(color_code[1]) # Set the hex value

    def validate_input(self):
        """Validates user input fields and returns parsed values or None if invalid."""
        try:
            start_x = float(self.start_x_var.get())
            start_y = float(self.start_y_var.get())
            end_x = float(self.end_x_var.get())
            end_y = float(self.end_y_var.get())
            iterations = int(self.iterations_var.get())
            offset = float(self.offset_var.get())
            roughness = float(self.roughness.get()) # Already a DoubleVar

            # Basic sanity checks
            if iterations < 0:
                raise ValueError("Iterations must be non-negative.")
            if offset < 0:
                raise ValueError("Offset size cannot be negative.")
            if not (0.0 < roughness <= 1.0): # Allow 1.0, ensure > 0
                 raise ValueError("Roughness must be between 0.0 (exclusive) and 1.0 (inclusive).")
            # Ensure start/end X are different to avoid division by zero in more complex displacement
            # if start_x == end_x:
            #     raise ValueError("Start X and End X must be different.")

            return start_x, start_y, end_x, end_y, iterations, offset, roughness

        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {e}\nPlease check the values entered.")
            return None

    def draw_command(self):
        """Handles the 'Draw' button click: validates input, generates points, draws polygon."""
        params = self.validate_input()
        if params is None:
            return # Stop if validation failed

        start_x, start_y, end_x, end_y, iterations, offset, roughness = params
        start_point = (start_x, start_y)
        end_point = (end_x, end_y)

        # --- Generate Fractal Points ---
        # Call the recursive function to get the intermediate points
        print(f"Generating terrain: Start={start_point}, End={end_point}, Iter={iterations}, Offset={offset}, Roughness={roughness}")
        fractal_points = generate_fractal_line(start_point, end_point, iterations, offset, roughness)
        print(f"Generated {len(fractal_points)} intermediate points.")

        # --- Prepare Polygon Points for Filling ---
        # The final list of points defining the top edge of the terrain
        terrain_line = [start_point] + fractal_points + [end_point]

        # Add points to close the polygon for filling the area *below* the line
        # Ensure points go from left-to-right along the top, then right-to-left along the bottom
        polygon_points = []
        polygon_points.extend(terrain_line)

        # Add bottom-right corner (extend vertically down from the end point)
        polygon_points.append((end_point[0], self.canvas_height))
        # Add bottom-left corner (extend vertically down from the start point)
        polygon_points.append((start_point[0], self.canvas_height))

        # Flatten the list of tuples for create_polygon
        flat_polygon_points = [coord for point in polygon_points for coord in point]

        # --- Draw on Canvas ---
        fill_color = self.selected_color.get()
        # Draw the filled polygon. Using the same outline color avoids a thin border.
        # Add a tag 'terrain_layer' to easily clear only these later if needed.
        self.canvas.create_polygon(
            flat_polygon_points,
            fill=fill_color,
            outline=fill_color, # Or use "black" or "" for different effects
            tags="terrain_layer"
        )
        print(f"Drawn polygon with {len(polygon_points)} vertices in color {fill_color}")

    def clear_command(self):
        """Clears the canvas of all drawings."""
        self.canvas.delete("all") # Delete all items on the canvas
        # Alternatively, if you only want to remove terrain layers:
        # self.canvas.delete("terrain_layer")
        print("Canvas cleared.")

# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = FractalTerrainApp(root)
    root.mainloop()