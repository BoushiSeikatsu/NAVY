import tkinter as tk
from tkinter import ttk, colorchooser, messagebox
import random
import math

# Function implements the core recursive midpoint displacement algorithm.
def generate_fractal_line(p1, p2, iterations, offset_size, roughness=0.5):
    # Base case: Stop recursion when no iterations are left.
    if iterations <= 0:
        return []

    # Calculate midpoint of the current segment.
    x1, y1 = p1
    x2, y2 = p2
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2

    # Apply random vertical displacement based on current offset size.
    displacement = random.uniform(-offset_size, offset_size)

    # Calculate the new displaced midpoint y-coordinate.
    displaced_mid_y = mid_y + displacement
    displaced_mid_point = (mid_x, displaced_mid_y)

    # Reduce offset for finer detail in the next recursive step, controlled by roughness.
    next_offset_size = offset_size * roughness

    # Recursive calls for the two new sub-segments (left and right).
    points_left = generate_fractal_line(p1, displaced_mid_point, iterations - 1, next_offset_size, roughness)
    points_right = generate_fractal_line(displaced_mid_point, p2, iterations - 1, next_offset_size, roughness)

    # Combine results: points from left recursion + the new midpoint + points from right recursion.
    return points_left + [displaced_mid_point] + points_right


# Main application class for the GUI, managing UI and interactions.
class FractalTerrainApp:
    # Initialize the main window and UI components.
    def __init__(self, master):
        self.master = master
        master.title("Fractal Terrain 2D Generator")

        # Define canvas dimensions and default configuration values.
        self.canvas_width = 900
        self.canvas_height = 600
        self.control_width = 200
        self.default_color = "#000000"
        # Use tk variables for dynamic UI updates.
        self.selected_color = tk.StringVar(value=self.default_color)
        self.roughness = tk.DoubleVar(value=0.6) # Default roughness value.

        # Setup main layout frames (canvas left, controls right).
        self.main_frame = ttk.Frame(master, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Frame for the drawing canvas.
        self.canvas_frame = ttk.Frame(self.main_frame)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Frame for control widgets. Fixed width.
        self.control_frame = ttk.Frame(self.main_frame, width=self.control_width, padding="10")
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, expand=False)
        self.control_frame.pack_propagate(False) # Prevent frame resizing based on content.

        # Create the main drawing canvas widget.
        self.canvas = tk.Canvas(self.canvas_frame, bg="white", width=self.canvas_width, height=self.canvas_height,
                                scrollregion=(0, 0, self.canvas_width, self.canvas_height)) # Scrollregion might be useful later.

        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Populate the control frame with widgets.
        self.setup_controls()

    # Create and arrange all control widgets (inputs, buttons, sliders).
    def setup_controls(self):
        frame = self.control_frame
        row_index = 0

        # Input fields for fractal generation parameters (start/end points).
        ttk.Label(frame, text="Start X (float):").grid(row=row_index, column=0, sticky="w", pady=2)
        self.start_x_var = tk.StringVar(value="0")
        ttk.Entry(frame, textvariable=self.start_x_var).grid(row=row_index, column=1, sticky="ew", pady=2)
        row_index += 1

        ttk.Label(frame, text="Start Y (float):").grid(row=row_index, column=0, sticky="w", pady=2)
        self.start_y_var = tk.StringVar(value=str(self.canvas_height / 2)) # Default Y to canvas middle.
        ttk.Entry(frame, textvariable=self.start_y_var).grid(row=row_index, column=1, sticky="ew", pady=2)
        row_index += 1

        ttk.Label(frame, text="End X (float):").grid(row=row_index, column=0, sticky="w", pady=2)
        self.end_x_var = tk.StringVar(value=str(self.canvas_width)) # Default X to canvas width.
        ttk.Entry(frame, textvariable=self.end_x_var).grid(row=row_index, column=1, sticky="ew", pady=2)
        row_index += 1

        ttk.Label(frame, text="End Y (float):").grid(row=row_index, column=0, sticky="w", pady=2)
        self.end_y_var = tk.StringVar(value=str(self.canvas_height / 2)) # Default Y to canvas middle.
        ttk.Entry(frame, textvariable=self.end_y_var).grid(row=row_index, column=1, sticky="ew", pady=2)
        row_index += 1

        ttk.Separator(frame, orient=tk.HORIZONTAL).grid(row=row_index, column=0, columnspan=2, sticky="ew", pady=5)
        row_index += 1

        # Input fields for fractal algorithm parameters (iterations, offset).
        ttk.Label(frame, text="Iterations (int):").grid(row=row_index, column=0, sticky="w", pady=2)
        self.iterations_var = tk.StringVar(value="8") # Default iterations.
        ttk.Entry(frame, textvariable=self.iterations_var).grid(row=row_index, column=1, sticky="ew", pady=2)
        row_index += 1

        ttk.Label(frame, text="Offset Size (float):").grid(row=row_index, column=0, sticky="w", pady=2)
        self.offset_var = tk.StringVar(value="50") # Initial displacement range.
        ttk.Entry(frame, textvariable=self.offset_var).grid(row=row_index, column=1, sticky="ew", pady=2)
        row_index += 1

        # Roughness scale for controlling terrain jaggedness (0.1 to 1.0).
        ttk.Label(frame, text="Roughness (0-1):").grid(row=row_index, column=0, sticky="w", pady=2)
        self.roughness_scale = ttk.Scale(frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL, variable=self.roughness)
        self.roughness_scale.grid(row=row_index, column=1, sticky="ew", pady=2)
        row_index += 2 # Adjusted index based on layout.

        ttk.Separator(frame, orient=tk.HORIZONTAL).grid(row=row_index, column=0, columnspan=2, sticky="ew", pady=5)
        row_index += 1

        # Color selection widgets.
        ttk.Label(frame, text="Selected Color:").grid(row=row_index, column=0, sticky="w", pady=2)
        self.color_display = tk.Label(frame, textvariable=self.selected_color, relief="sunken", width=10,
                                      bg=self.selected_color.get()) # Visual preview of the color.
        self.color_display.grid(row=row_index, column=1, sticky="ew", pady=2)
        # Trace variable changes to update the color preview.
        self.selected_color.trace_add("write", self.update_color_display)
        row_index += 1

        self.pick_color_button = ttk.Button(frame, text="Pick Color", command=self.pick_color_command)
        self.pick_color_button.grid(row=row_index, column=0, columnspan=2, sticky="ew", pady=5)
        row_index += 1

        ttk.Separator(frame, orient=tk.HORIZONTAL).grid(row=row_index, column=0, columnspan=2, sticky="ew", pady=5)
        row_index += 1

        # Buttons for drawing the terrain and clearing the canvas.
        self.draw_button = ttk.Button(frame, text="Draw Terrain Layer", command=self.draw_command)
        self.draw_button.grid(row=row_index, column=0, columnspan=2, sticky="ew", pady=5)
        row_index += 1

        self.clear_button = ttk.Button(frame, text="Clear Canvas", command=self.clear_command)
        self.clear_button.grid(row=row_index, column=0, columnspan=2, sticky="ew", pady=5)
        row_index += 1

        # Allow the second column (widgets) to expand horizontally if needed.
        frame.columnconfigure(1, weight=1)

    # Callback function to update the color preview label's background.
    def update_color_display(self, *args):
        try:
            self.color_display.config(bg=self.selected_color.get())
        except tk.TclError: # Handles potential brief invalid color name during typing.
            self.color_display.config(bg=self.default_color) # Revert to default if error.

    # Opens the standard Tkinter color chooser dialog.
    def pick_color_command(self):
        # Askcolor returns a tuple ((r,g,b), hex_string) or (None, None).
        color_code = colorchooser.askcolor(title="Choose Terrain Color", initialcolor=self.selected_color.get())
        # Check if a color was selected and a hex string is available.
        if color_code and color_code[1]:
            self.selected_color.set(color_code[1]) # Update the bound variable.

    # Validates user input values from the entry fields. Important for stability.
    def validate_input(self):
        try:
            # Attempt to convert string inputs to appropriate numeric types.
            start_x = float(self.start_x_var.get())
            start_y = float(self.start_y_var.get())
            end_x = float(self.end_x_var.get())
            end_y = float(self.end_y_var.get())
            iterations = int(self.iterations_var.get())
            offset = float(self.offset_var.get())
            roughness = float(self.roughness.get()) # Get value from DoubleVar.

            # Basic range checks for algorithm parameters.
            if iterations < 0:
                raise ValueError("Iterations must be non-negative.")
            if offset < 0:
                raise ValueError("Offset size cannot be negative.")
            if not (0.0 < roughness <= 1.0): # Ensure roughness is within valid range.
                 raise ValueError("Roughness must be between 0.0 (exclusive) and 1.0 (inclusive).")

            # Return parsed values if valid.
            return start_x, start_y, end_x, end_y, iterations, offset, roughness

        # Catch conversion errors (e.g., non-numeric input) or range errors.
        except ValueError as e:
            # Display error message to the user.
            messagebox.showerror("Input Error", f"Invalid input: {e}\nPlease check the values entered.")
            return None # Indicate validation failure.

    # Handles the 'Draw Terrain Layer' button action.
    def draw_command(self):
        # First, validate the user's input parameters.
        params = self.validate_input()
        # If validation fails (returns None), stop execution.
        if params is None:
            return

        # Unpack the validated parameters.
        start_x, start_y, end_x, end_y, iterations, offset, roughness = params
        start_point = (start_x, start_y)
        end_point = (end_x, end_y)

        # Generate the list of fractal points using the recursive function.
        print(f"Generating terrain: Start={start_point}, End={end_point}, Iter={iterations}, Offset={offset}, Roughness={roughness}")
        fractal_points = generate_fractal_line(start_point, end_point, iterations, offset, roughness)
        print(f"Generated {len(fractal_points)} intermediate points.")

        # Combine start, generated points, and end point for the top terrain line.
        terrain_line = [start_point] + fractal_points + [end_point]

        # Prepare points list for polygon filling (terrain area below the line).
        polygon_points = []
        polygon_points.extend(terrain_line) # Add the top line points.

        # Add bottom-right corner (directly below the end point at canvas bottom).
        polygon_points.append((end_point[0], self.canvas_height))
        # Add bottom-left corner (directly below the start point at canvas bottom).
        polygon_points.append((start_point[0], self.canvas_height))
        # This order creates a closed shape for filling.

        # Flatten the list of (x,y) tuples into a single list [x1, y1, x2, y2, ...] for create_polygon.
        flat_polygon_points = [coord for point in polygon_points for coord in point]

        # Get the currently selected fill color.
        fill_color = self.selected_color.get()

        # Draw the filled polygon on the canvas.
        # Outline is set to fill color for a solid appearance.
        # A tag is added to potentially clear only these layers later.
        self.canvas.create_polygon(
            flat_polygon_points,
            fill=fill_color,
            outline=fill_color, # Setting outline=fill_color avoids visible border.
            tags="terrain_layer" # Tag allows selective deletion if needed.
        )
        print(f"Drawn polygon with {len(polygon_points)} vertices in color {fill_color}")

    # Handles the 'Clear Canvas' button action.
    def clear_command(self):
        # Delete all items currently drawn on the canvas.
        self.canvas.delete("all")
        # Could alternatively use self.canvas.delete("terrain_layer") to only remove tagged items.
        print("Canvas cleared.")


# Standard Python entry point to run the application.
if __name__ == "__main__":
    # Create the main Tkinter window.
    root = tk.Tk()
    # Instantiate the application class.
    app = FractalTerrainApp(root)
    # Start the Tkinter event loop.
    root.mainloop()