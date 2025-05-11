import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- System Parameters ---
L1 = 1.0  # Length of the first pendulum arm (meters)
L2 = 1.0  # Length of the second pendulum arm (meters)
M1 = 1.0  # Mass of the first bob (kilograms)
M2 = 1.0  # Mass of the second bob (kilograms)
G = 9.81  # Acceleration due to gravity (m/s^2)


# theta1_0: initial angle of the first pendulum (radians)
# omega1_0: initial angular velocity of the first pendulum (radians/s)
# theta2_0: initial angle of the second pendulum (radians)
# omega2_0: initial angular velocity of the second pendulum (radians/s)
theta1_0 = np.pi / 3  # (2*pi/6 from PDF)
omega1_0 = 0.0
theta2_0 = 5 * np.pi / 8
omega2_0 = 0.0

# Initial state vector: [theta1, omega1, theta2, omega2]
y0 = np.array([theta1_0, omega1_0, theta2_0, omega2_0])

# --- Time Array for Simulation ---
t_max = 20.0  # Total simulation time (seconds)
dt = 0.01    # Time step for ODE solver (seconds)
t_points = np.arange(0, t_max, dt)

# --- Equations of Motion ---
# This function defines the system of first-order ODEs.
# state = [theta1, omega1, theta2, omega2]
# t = time
# L1, L2, M1, M2, G are system parameters.
# Returns: [d(theta1)/dt, d(omega1)/dt, d(theta2)/dt, d(omega2)/dt]
def get_derivative(state, t, l1, l2, m1, m2, g):
    """
    Calculates the derivatives of the state variables for the double pendulum.
    The state is [theta1, omega1, theta2, omega2].
    The equations for angular accelerations (alpha1, alpha2) are from page 6 of the PDF.
    """
    th1, w1, th2, w2 = state

    # Differences in angles and their trigonometric functions
    dth = th1 - th2
    s_dth = np.sin(dth)
    c_dth = np.cos(dth)
    s_th1 = np.sin(th1)
    s_th2 = np.sin(th2)

    # Derivatives:
    # d(theta1)/dt = omega1
    # d(theta2)/dt = omega2
    dth1_dt = w1
    dth2_dt = w2

    # Angular acceleration for the first pendulum (alpha1 or theta1_ddot)
    # num_alpha1 = m2*g*sin(th2)*cos(th1-th2) - m2*sin(th1-th2)*(l1*w1^2*cos(th1-th2) + l2*w2^2) - (m1+m2)*g*sin(th1)
    # den_alpha1 = l1*(m1 + m2*sin(th1-th2)^2)
    num_alpha1 = m2 * g * s_th2 * c_dth - \
                 m2 * s_dth * (l1 * w1**2 * c_dth + l2 * w2**2) - \
                 (m1 + m2) * g * s_th1
    den_alpha1 = l1 * (m1 + m2 * s_dth**2)
    dw1_dt = num_alpha1 / den_alpha1  # alpha1

    # Angular acceleration for the second pendulum (alpha2 or theta2_ddot)
    # num_alpha2 = (m1+m2)*(l1*w1^2*sin(th1-th2) - g*sin(th2) + g*sin(th1)*cos(th1-th2)) + m2*l2*w2^2*sin(th1-th2)*cos(th1-th2)
    # den_alpha2 = l2*(m1 + m2*sin(th1-th2)^2)
    num_alpha2 = (m1 + m2) * (l1 * w1**2 * s_dth - g * s_th2 + g * s_th1 * c_dth) + \
                 m2 * l2 * w2**2 * s_dth * c_dth
    den_alpha2 = l2 * (m1 + m2 * s_dth**2)
    dw2_dt = num_alpha2 / den_alpha2  # alpha2
    
    return [dth1_dt, dw1_dt, dth2_dt, dw2_dt]

# --- Solve the ODE System ---
print(f"Solving ODEs for {t_max}s with dt={dt}s...")
sol = odeint(get_derivative, y0, t_points, args=(L1, L2, M1, M2, G))
print(f"ODEs solved. Total simulation data points: {len(sol)}")

# Extract angular positions from solution
theta1_sim = sol[:, 0]
theta2_sim = sol[:, 2]

# --- Convert to Cartesian Coordinates for all simulated points ---
# x1 = l1 * sin(theta1)
# y1 = -l1 * cos(theta1) (negative for y-axis pointing upwards from origin at pivot)
# x2 = x1 + l2 * sin(theta2)
# y2 = y1 - l2 * cos(theta2)
x1_coords_sim = L1 * np.sin(theta1_sim)
y1_coords_sim = -L1 * np.cos(theta1_sim)
x2_coords_sim = L1 * np.sin(theta1_sim) + L2 * np.sin(theta2_sim)
y2_coords_sim = -L1 * np.cos(theta1_sim) - L2 * np.cos(theta2_sim)

# --- Animation Setup ---
print("Setting up animation...")

# GIF parameters
gif_fps = 25  # Desired frames per second for the output GIF

# Determine the step for selecting frames from the simulation data for animation
# This ensures the animation duration matches t_max when played at gif_fps
sim_rate = 1.0 / dt  # Simulation data points per second
animation_step = int(max(1, round(sim_rate / gif_fps)))

# Number of frames for the FuncAnimation (and thus for the final GIF)
num_animation_frames = int(len(t_points) / animation_step)

print(f"Simulation data rate: {sim_rate:.0f} Hz")
print(f"Target GIF FPS: {gif_fps} fps")
print(f"Animation will use every {animation_step}-th frame from simulation data.")
print(f"Total frames in GIF: {num_animation_frames}")


fig = plt.figure(figsize=(8.5, 8.5))
ax_limit = (L1 + L2) * 1.1 # Ensure pendulum fits in the view
ax = fig.add_subplot(autoscale_on=False,
                     xlim=(-ax_limit, ax_limit),
                     ylim=(-ax_limit, ax_limit))
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_xlabel("x (meters)")
ax.set_ylabel("y (meters)")
ax.set_title("Chaotic Motion of a Double Pendulum")

# Animation elements
line1, = ax.plot([], [], 'o-', lw=2, markersize=8, color='dodgerblue', markevery=[1], markerfacecolor='lightblue') # Bob 1
line2, = ax.plot([], [], 'o-', lw=2, markersize=8, color='orangered', markevery=[1], markerfacecolor='lightsalmon')  # Bob 2
pivot, = ax.plot([0], [0], 'o', markersize=7, color='black') # Pivot point

time_template = 'Time = %.1fs'
time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray', boxstyle='round,pad=0.3'))

# Trace for the second bob's path
trace_length_sim_steps = 200 # Length of the trace in terms of original simulation steps
trace_line, = ax.plot([], [], '-', lw=1, color='grey', alpha=0.7)


def init_animation():
    """Initializes the animation elements for a clean slate."""
    line1.set_data([], [])
    line2.set_data([], [])
    trace_line.set_data([], [])
    time_text.set_text('')
    return line1, line2, trace_line, time_text, pivot

def animate(anim_frame_idx):
    """Performs animation step for a given animation frame index."""
    # Map the animation frame index to the actual simulation data index
    sim_data_idx = anim_frame_idx * animation_step
    
    # Safeguard, though num_animation_frames should prevent out-of-bounds
    if sim_data_idx >= len(t_points):
        sim_data_idx = len(t_points) - 1

    # Origin (pivot point)
    x0, y0 = 0, 0

    # Get current Cartesian coordinates from the pre-calculated simulation arrays
    current_x1 = x1_coords_sim[sim_data_idx]
    current_y1 = y1_coords_sim[sim_data_idx]
    current_x2 = x2_coords_sim[sim_data_idx]
    current_y2 = y2_coords_sim[sim_data_idx]

    # Update pendulum arm positions
    line1.set_data([x0, current_x1], [y0, current_y1])
    line2.set_data([current_x1, current_x2], [current_y1, current_y2])

    # Update time display using the actual time from the simulation point
    time_text.set_text(time_template % (t_points[sim_data_idx]))
    
    # Update trace for bob 2 using the dense simulation data for a smooth trail
    trace_actual_end_idx = sim_data_idx + 1
    trace_actual_start_idx = max(0, trace_actual_end_idx - trace_length_sim_steps)
    
    trace_line.set_data(x2_coords_sim[trace_actual_start_idx:trace_actual_end_idx],
                        y2_coords_sim[trace_actual_start_idx:trace_actual_end_idx])

    return line1, line2, trace_line, time_text, pivot

# --- Create and Save Animation ---
# Interval for FuncAnimation: delay between frames in milliseconds.
# This should align with the target GIF FPS.
gif_interval = 1000 / gif_fps

# Create the animation object
ani = animation.FuncAnimation(fig,
                              animate,
                              frames=num_animation_frames, # Total number of frames for the animation
                              interval=gif_interval,       # Delay between frames in ms
                              blit=True,                   # Optimize drawing
                              init_func=init_animation)    # Function to draw a clear frame

# Save the animation as a GIF
output_filename = 'double_pendulum_chaotic_motion.gif'
print(f"Saving animation to {output_filename} ({num_animation_frames} frames at {gif_fps} FPS)...")
print("This process might take some time depending on the animation length and system performance.")
try:
    # Attempt to use ImageMagick, ensure fps is specified
    print(f"Saving animation to {output_filename} with ImageMagick ({num_animation_frames} frames at {gif_fps} FPS)...")
    ani.save(output_filename, writer='imagemagick', fps=gif_fps)
    print(f"Animation saved successfully to {output_filename}")
except Exception as e_imagemagick:
    print(f"Error saving animation with ImageMagick: {e_imagemagick}")
    print("Make sure ImageMagick is installed and in your system's PATH.")
    print("Falling back to Pillow...")
    try:
        ani.save(output_filename, writer='pillow', fps=gif_fps)
        print(f"Animation saved successfully to {output_filename} using Pillow.")
    except Exception as e_pillow:
        print(f"Error saving animation with Pillow as well: {e_pillow}")
        print("Please check Pillow installation and potential image complexity issues.")
