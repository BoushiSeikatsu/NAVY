import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# We'll use pillow for saving the GIFs, make sure it's installed!
# import numpy as np # Still not strictly needed, but good to keep in mind for complex stuff

# --- Transformation Data ---

# Here are the numbers defining the transformations for the first fern thingy.
# Each row is like a recipe: [a, b, c, d, e, f, g, h, i, j, k, l]
model1_params = [
    [0.00, 0.00, 0.01, 0.00, 0.26, 0.00, 0.00, 0.00, 0.05, 0.00, 0.00, 0.00], # Stem-like part
    [0.20,-0.26,-0.01, 0.23, 0.22,-0.07, 0.07, 0.00, 0.24, 0.00, 0.80, 0.00], # Smaller leaflets leaning left
    [-0.25, 0.28, 0.01, 0.26, 0.24,-0.07, 0.07, 0.00, 0.24, 0.00, 0.22, 0.00], # Smaller leaflets leaning right
    [0.85, 0.04,-0.01,-0.04, 0.85, 0.09, 0.00, 0.08, 0.84, 0.00, 0.80, 0.00]  # Successively smaller copies overall
]

# And here's the recipe book for the second funky fractal model.
# Same format: [a, b, c, d, e, f, g, h, i, j, k, l]
model2_params = [
    [0.05, 0.00, 0.00, 0.00, 0.60, 0.00, 0.00, 0.00, 0.05, 0.00, 0.00, 0.00], # Main stem/trunk shrink and lift
    [0.45,-0.22, 0.22, 0.22, 0.45, 0.22,-0.22, 0.22,-0.45, 0.00, 1.00, 0.00], # Branching structure 1
    [-0.45, 0.22,-0.22, 0.22, 0.45, 0.22, 0.22,-0.22, 0.45, 0.00, 1.25, 0.00], # Branching structure 2
    [0.49,-0.08, 0.08, 0.08, 0.49, 0.08, 0.08,-0.08, 0.49, 0.00, 2.00, 0.00]  # Smaller details/twigs
]

# --- IFS Generation Function ---

def generate_ifs_points(transformations, num_points, initial_point=(0, 0, 0)):
    """
    Okay, this function does the heavy lifting. It takes the transformation rules,
    how many points we want, and where to start, then spits out a list of
    all the points making up the fractal. It's like a chaotic dot-to-dot.

    Args:
        transformations (list): The list of parameter lists (our recipes).
        num_points (int): How many dots we want on our picture.
        initial_point (tuple): The (x, y, z) coordinates to kick things off.

    Returns:
        list: A list of (x, y, z) tuples, basically the fractal's skeleton.
    """
    # Get our starting position
    x, y, z = initial_point
    # We need to remember where we've been to draw it later. Start with the first spot.
    points_history = [initial_point]

    # How many different ways can we transform the point?
    num_transformations = len(transformations)
    # Since each transformation has the same chance (0.25), we just need
    # a list of the possible choices (like rolling a 4-sided die).
    transformation_indices = list(range(num_transformations))

    print(f"Alright, let's generate {num_points} points... this might take a sec.")
    # Loop for the specified number of points (iterations)
    for i in range(num_points):
        # Pick one of the transformation recipes at random.
        choice_index = random.choice(transformation_indices)
        params = transformations[choice_index]

        # Grab the parameters from the chosen recipe for easier reading.
        a, b, c, d, e, f, g, h, i, j, k, l = params

        # Apply the magic math (affine transformation) to get the next point's location.
        # x_new = a*x_old + b*y_old + c*z_old + j
        # y_new = d*x_old + e*y_old + f*z_old + k
        # z_new = g*x_old + h*y_old + i*z_old + l
        x_new = a*x + b*y + c*z + j
        y_new = d*x + e*y + f*z + k
        z_new = g*x + h*y + i*z + l

        # The new point becomes our current point for the next round.
        x, y, z = x_new, y_new, z_new

        # Add this newly calculated point to our list of historical points.
        points_history.append((x, y, z))

        # Let's give a little progress update so you know it's not crashed!
        if (i + 1) % (num_points // 20) == 0: # Update every 5%
           print(f"  ...{i + 1}/{num_points} points generated...")

    print("Done generating points! Phew.")
    return points_history

# --- Animation Function ---

def animate_fractal(points, title="IFS Fractal", filename="fractal_animation.gif", num_frames=200, interval=50):
    """
    Takes the list of points and makes a cool 3D animation showing
    the fractal appearing bit by bit. Saves it as a GIF.

    Args:
        points (list): The list of (x, y, z) points from generate_ifs_points.
        title (str): What to call the plot window.
        filename (str): The name for the output GIF file.
        num_frames (int): How many frames the animation should have. More frames = smoother, larger file.
        interval (int): How many milliseconds between frames (controls speed). Lower = faster.
    """
    if not points:
        print("Hey, there are no points to animate!")
        return

    print(f"Setting up animation: {title}")

    # Need to split the (x,y,z) tuples into separate lists for plotting
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    z_coords = [p[2] for p in points]

    # Total points we have
    total_points = len(points)
    # How many new points to add in each frame of the animation
    points_per_frame = total_points // num_frames

    # Set up the 3D plot stage
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Set the plot limits based on the range of all points, so it doesn't jump around
    ax.set_xlim(min(x_coords) - 0.1, max(x_coords) + 0.1)
    ax.set_ylim(min(y_coords) - 0.1, max(y_coords) + 0.1)
    ax.set_zlim(min(z_coords) - 0.1, max(z_coords) + 0.1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    # Initialize an empty scatter plot. We'll update this in the animation.
    # Using small markers and slight transparency looks nice.
    scatter = ax.scatter([], [], [], s=1, marker='.', alpha=0.7)

    # This function gets called for each frame of the animation
    def update(frame):
        # Calculate how many points should be visible in this frame
        current_point_count = min((frame + 1) * points_per_frame, total_points)
        # Get the data for the points up to this frame
        current_x = x_coords[:current_point_count]
        current_y = y_coords[:current_point_count]
        current_z = z_coords[:current_point_count]

        # Update the scatter plot data. Note: _offsets3d is the way to update 3D scatter data
        scatter._offsets3d = (current_x, current_y, current_z)

        # Update the title to show progress (optional)
        ax.set_title(f"{title} (Frame {frame+1}/{num_frames})")

        # We need to return the plot object that got changed
        return scatter,

    print(f"Creating animation ({num_frames} frames)... This is the slow part!")
    # Create the animation object
    ani = FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=True) # blit=True helps performance slightly

    # Save the animation as a GIF
    print(f"Saving animation to {filename}...")
    try:
        # You need 'pillow' installed for this writer! (pip install pillow)
        ani.save(filename, writer='pillow', fps=1000 // interval)
        print(f"Successfully saved {filename}")
    except Exception as e:
        print(f"\n*** Oh no, couldn't save the GIF! Error: {e} ***")
        print("*** Did you install pillow? Try: pip install pillow ***")


    # We don't need plt.show() anymore since we're saving the file.
    # If you want to see the plot window *after* saving, uncomment the next line
    # plt.show()
    # Close the plot figure to free up memory
    plt.close(fig)


# --- Main Execution ---

if __name__ == "__main__":
    # How many points should we generate? More points = clearer fractal, but slower generation & bigger GIFs.
    # 50k is a decent starting point for testing. Try 100k or more for nicer results if you have time.
    NUM_ITERATIONS = 50000
    # How many frames for the GIF animation?
    ANIMATION_FRAMES = 150
    # How fast the animation plays (milliseconds per frame)
    ANIMATION_INTERVAL = 40 # Lower is faster

    # --- Generate and Animate First Model ---
    print("--- Processing First Model ---")
    points_model1 = generate_ifs_points(model1_params, NUM_ITERATIONS)
    animate_fractal(points_model1,
                    title="First Model Build-up",
                    filename="first_model_fractal.gif",
                    num_frames=ANIMATION_FRAMES,
                    interval=ANIMATION_INTERVAL)

    # --- Generate and Animate Second Model ---
    print("\n--- Processing Second Model ---")
    points_model2 = generate_ifs_points(model2_params, NUM_ITERATIONS)
    animate_fractal(points_model2,
                    title="Second Model Build-up",
                    filename="second_model_fractal.gif",
                    num_frames=ANIMATION_FRAMES,
                    interval=ANIMATION_INTERVAL)

    print("\nAll done! Check your folder for the GIF files.")