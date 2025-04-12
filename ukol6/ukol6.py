import math
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time # Gotta measure how long things take!

# -------------------------------------
# L-System String Generator
# -------------------------------------
def generate_lsystem(axiom, rules, iterations):
    """
    Whips up the final L-system string by applying rules over several iterations.
    It's like recursive growth, but with strings!
    """
    current_string = axiom
    print("Generating L-system string:")
    print(f"  Starting point (Iteration 0): {len(current_string)} chars")
    for i in range(iterations):
        start_time = time.time()
        # Let's build the next version of the string
        # Using list comprehension might be a tad quicker when strings get huge
        next_string = "".join([rules.get(char, char) for char in current_string])
        current_string = next_string
        end_time = time.time()
        print(f"  After iteration {i+1}: {len(current_string)} chars long (took {end_time - start_time:.2f}s)")
        # Uncomment below if things get crazy long and you need to cap it
        # max_len = 500000
        # if len(current_string) > max_len:
        #     print(f"Whoa, string length hit {max_len}! Trimming it down.")
        #     current_string = current_string[:max_len]
        #     break
    return current_string

# -------------------------------------
# Figuring out where the lines go
# -------------------------------------
def calculate_lines(instructions, angle_degrees, length, start_pos=(0, 0), start_angle_degrees=90):
    """
    Takes the L-system recipe (instructions) and figures out all the
    start and end points for the lines we need to draw. Simulates the turtle
    without actually drawing yet.
    """
    lines = [] # This will hold all our line segments: ((x1, y1), (x2, y2))
    x, y = start_pos # Where we start drawing
    current_angle_degrees = start_angle_degrees # Which way we're facing initially
    state_stack = [] # For handling the '[' and ']' branching stuff

    print(f"Figuring out lines for {len(instructions)} instructions...")
    count = 0
    f_commands = instructions.count('F') # Let's see how many 'F's we have to process
    start_time = time.time()

    for command in instructions:
        if command == 'F': # Move forward and draw
            # Convert angle to radians for math functions
            rad = math.radians(current_angle_degrees)
            # Calculate where the line ends
            x2 = x + length * math.cos(rad)
            y2 = y + length * math.sin(rad)
            # Add this line segment to our list
            lines.append(((x, y), (x2, y2)))
            # Update our current position to the end of the line
            x, y = x2, y2
            count += 1
            if count % 50000 == 0: # Print a little progress update for big jobs
                 elapsed = time.time() - start_time
                 print(f"  Processed {count}/{f_commands} 'F' commands... ({elapsed:.1f}s)")
        elif command == 'b': # Move forward, but keep the pen up (no drawing)
            rad = math.radians(current_angle_degrees)
            x += length * math.cos(rad)
            y += length * math.sin(rad)
        elif command == '+': # Turn right (decrease angle)
            current_angle_degrees -= angle_degrees
        elif command == '-': # Turn left (increase angle)
            current_angle_degrees += angle_degrees
        elif command == '[': # Save current position and angle (like a checkpoint)
            state_stack.append({'x': x, 'y': y, 'angle': current_angle_degrees})
        elif command == ']': # Go back to the last saved checkpoint
            if state_stack: # Make sure there's something saved!
                state = state_stack.pop()
                x = state['x']
                y = state['y']
                current_angle_degrees = state['angle']

    end_time = time.time()
    print(f"Done calculating! Got {len(lines)} line segments in {end_time - start_time:.2f}s.")
    return lines

# -------------------------------------
# Make the animation and save the GIF (Drawing in chunks)
# -------------------------------------
def animate_lsystem_matplotlib_chunked(lines, filename, chunk_size=100, interval=50, line_color='black', bg_color='white'):
    """
    Uses Matplotlib to create an animation by drawing the calculated lines
    in batches (chunks) for better performance, then saves it as a GIF.

    Args:
        lines: The list of line segments from calculate_lines.
        filename: Where to save the GIF.
        chunk_size: How many new lines to draw in each frame of the animation.
        interval: How long to pause between frames (milliseconds).
        line_color: What color the lines should be.
        bg_color: Background color for the plot.
    """
    if not lines: # If there's nothing to draw, don't bother
        print(f"Warning: No lines generated for {filename}. Skipping animation.")
        return
    if chunk_size <= 0: # Chunk size must be at least 1
        chunk_size = 1

    # Set up the plotting area
    fig, ax = plt.subplots()
    fig.patch.set_facecolor(bg_color) # Set background for the figure area
    ax.set_facecolor(bg_color)      # Set background for the plot area itself

    # Figure out the drawing boundaries based on all line points
    # We only need to do this once
    min_x, min_y = lines[0][0]
    max_x, max_y = lines[0][0]
    for (x1, y1), (x2, y2) in lines:
        min_x = min(min_x, x1, x2)
        max_x = max(max_x, x1, x2)
        min_y = min(min_y, y1, y2)
        max_y = max(max_y, y1, y2)

    # Add a bit of padding around the edges so the drawing isn't squished
    padding_x = (max_x - min_x) * 0.05 + 1 # Ensure at least 1 unit padding
    padding_y = (max_y - min_y) * 0.05 + 1
    ax.set_xlim(min_x - padding_x, max_x + padding_x)
    ax.set_ylim(min_y - padding_y, max_y + padding_y)
    # Make sure 1 unit on the x-axis looks the same length as 1 unit on the y-axis
    ax.set_aspect('equal', adjustable='box')
    # Hide the messy axes labels and ticks
    ax.axis('off')

    # How many frames will the final animation have?
    num_total_lines = len(lines)
    num_frames = math.ceil(num_total_lines / chunk_size)

    print(f"Setting up animation: {num_frames} frames total (drawing {chunk_size} lines per frame)...")

    # This function gets called for each frame of the animation
    def update(frame_num):
        # Figure out which lines to draw up to for this frame
        last_line_index = min((frame_num + 1) * chunk_size, num_total_lines)

        # Clear the previous frame's drawing completely
        # (Could try optimizing by only drawing *new* lines, but clear is safer)
        ax.clear()
        # Reapply settings that got cleared
        ax.set_facecolor(bg_color)
        ax.set_xlim(min_x - padding_x, max_x + padding_x)
        ax.set_ylim(min_y - padding_y, max_y + padding_y)
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

        # Draw all the lines from the beginning up to the current limit
        for i in range(last_line_index):
             (x1, y1), (x2, y2) = lines[i]
             # Draw the actual line segment
             ax.plot([x1, x2], [y1, y2], color=line_color, lw=0.8) # Linewidth 0.8 looks okay

        # Add a title to show progress (optional, but helpful)
        ax.set_title(f"Drawing lines up to {last_line_index}/{num_total_lines}")
        # Print progress to console too, but not too often
        if frame_num % 20 == 0 or frame_num == num_frames - 1:
             print(f"  Rendering frame {frame_num+1}/{num_frames}")
        # Important: FuncAnimation needs this function to return the changed plot elements
        return ax,

    # Create the animation object. This doesn't save yet, just sets it up.
    anim = animation.FuncAnimation(fig, update, frames=num_frames,
                                   interval=interval, blit=False) # blit=False tends to be more reliable

    # Now, actually save the animation to a file
    print(f"Saving animation to {filename} (this might take a while)...")
    start_save_time = time.time()
    try:
        # Use 'pillow' writer for GIFs. Adjust fps and dpi as needed.
        # Higher dpi = better quality = bigger file & longer save time.
        anim.save(filename, writer='pillow', fps=max(1, 1000 // interval), dpi=100)
        end_save_time = time.time()
        print(f"Success! Saved {filename} (took {end_save_time - start_save_time:.2f}s)")
    except Exception as e:
        end_save_time = time.time()
        print(f"Oops, error saving animation: {e} (failed after {end_save_time - start_save_time:.2f}s)")
        print("Maybe double-check if Matplotlib and Pillow are installed okay?")

    plt.close(fig) # Close the plot figure to free up memory

# -------------------------------------
# Main part - where we run everything
# -------------------------------------
if __name__ == "__main__":

    # Where should we save the final GIFs?
    output_dir = "lsystem_gifs_matplotlib"
    # Create the directory if it doesn't exist yet
    os.makedirs(output_dir, exist_ok=True)

    # Define the parameters for the L-systems we want to draw
    lsystems = [
        {
            "name": "Quadratic_Koch",
            "axiom": "F+F+F+F",
            "rules": {"F": "F+F-F-FF+F+F-F"}, # This rule grows FAST
            "angle": 90,
            "iterations": 2, # Reduced from 3, otherwise it gets HUGE
            "length": 8,     # Made lines slightly longer to compensate
            "start_pos": (0, 0), # Start at the origin
            "start_heading": 90, # Pointing up
            "interval": 300,      # Animation speed (milliseconds per frame)
            "chunk_size": 30    # How many lines per frame
        },
        {
            "name": "Koch_Snowflake_Variant",
            "axiom": "F++F++F",
            "rules": {"F": "F+F--F+F"},
            "angle": 60,
            "iterations": 4, # This one's less crazy, 4 iterations is usually fine
            "length": 8,
            "start_pos": (0, 0),
            "start_heading": 0, # Start pointing right this time
            "interval": 30,
            "chunk_size": 30
        },
        {
            "name": "Fractal_Plant_1",
            "axiom": "F",
            "rules": {"F": "F[+F]F[-F]F"}, # Classic branching plant
            "angle": math.degrees(math.pi / 7), # Angle needs conversion to degrees
            "iterations": 5,
            "length": 6,
            "start_pos": (0, 0),
            "start_heading": 90, # Pointing up
            "interval": 50,
            "chunk_size": 100
        },
        {
            "name": "Fractal_Plant_2",
            "axiom": "F",
            "rules": {"F": "FF+[+F-F-F]-[-F+F+F]"}, # Another plant-like fractal
            "angle": math.degrees(math.pi / 8),
            "iterations": 4,
            "length": 10,
            "start_pos": (0, 0),
            "start_heading": 90,
            "interval": 50,
            "chunk_size": 150 # Adjusted chunk size
        }
    ]

    # Let's loop through our list of L-systems and process each one
    total_start_time = time.time()
    for i, system in enumerate(lsystems):
        print(f"\n--- Processing L-System {i+1}: {system['name']} ---")
        system_start_time = time.time()

        # Generate the instruction string
        instructions = generate_lsystem(system['axiom'], system['rules'], system['iterations'])

        # Calculate all the line segments needed
        lines = calculate_lines(
            instructions,
            system['angle'],
            system['length'],
            start_pos=system['start_pos'],
            start_angle_degrees=system['start_heading']
        )

        # Create the animation and save it as a GIF
        gif_filename = os.path.join(output_dir, f"lsystem_{i+1}_{system['name']}_mpl_chunked.gif")
        animate_lsystem_matplotlib_chunked(
            lines,
            gif_filename,
            chunk_size=system.get('chunk_size', 200), # Use defined chunk_size or default to 200
            interval=system.get('interval', 50)     # Use defined interval or default to 50
        )
        system_end_time = time.time()
        print(f"--- System {i+1} processing finished in {system_end_time - system_start_time:.2f}s ---")

    total_end_time = time.time()
    print("\nAll done! L-system GIFs should be in the '{}' folder.".format(output_dir))
    print(f"Total time for everything: {total_end_time - total_start_time:.2f}s")