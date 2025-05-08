import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import random

# Define the states of a cell
EMPTY = 0
TREE = 1
BURNING = 2
BURNT = 3  # A cell that has finished burning, as per Rule 4 ("stay burned")

# Define simulation parameters as per the example
GRID_SIZE = 100  # Grid dimensions (100x100)
P_REGROWTH = 0.05  # Probability of new tree growth (p)
F_IGNITION = 0.001 # Probability of spontaneous ignition (f)
INITIAL_FOREST_DENSITY = 0.5 # Initial proportion of cells that are trees

# Neighborhood type: 'von_neumann' or 'moore'
# The problem suggests von Neumann as default.
NEIGHBORHOOD_TYPE = 'von_neumann'


class ForestFireAutomaton:
    """
    A Cellular Automaton for simulating a forest fire.

    The simulation follows these rules:
    1. An empty area (EMPTY) or a burnt tree (BURNT) is replaced with probability `p`
       by a live tree (TREE). If not replaced by a tree, it becomes/remains EMPTY.
    2. A tree (TREE) will catch fire (BURNING) if any of its neighbors are on fire.
    3. If no neighbor of a tree (TREE) is burning, it will start burning (BURNING)
       with a small probability `f` (e.g., due to a lightning strike).
    4. A burning tree (BURNING) will burn out in the next time step and become
       a burnt tree (BURNT).
    """

    def __init__(self, size, p_regrowth, f_ignition, initial_density, neighborhood_type='von_neumann'):
        """
        Initialize the forest fire simulation.

        Args:
            size (int): The dimension of the square grid (size x size).
            p_regrowth (float): Probability of a new tree growing in an EMPTY or BURNT cell.
            f_ignition (float): Probability of a tree spontaneously catching fire.
            initial_density (float): Initial proportion of cells that are trees.
            neighborhood_type (str): 'von_neumann' or 'moore'.
        """
        self.size = size
        self.p = p_regrowth
        self.f = f_ignition
        self.initial_density = initial_density
        self.neighborhood_type = neighborhood_type.lower()

        if self.neighborhood_type not in ['von_neumann', 'moore']:
            raise ValueError("neighborhood_type must be 'von_neumann' or 'moore'")

        self.grid = self._initialize_grid()

    def _initialize_grid(self):
        """
        Create the initial grid state.
        Cells are randomly set to TREE based on initial_density, otherwise EMPTY.
        """
        grid = np.zeros((self.size, self.size), dtype=int)
        for r in range(self.size):
            for c in range(self.size):
                if random.random() < self.initial_density:
                    grid[r, c] = TREE
                else:
                    grid[r, c] = EMPTY
        return grid

    def _get_neighbors(self, r, c):
        """
        Get the coordinates of valid neighbors for a cell (r, c).
        Handles boundary conditions (neighbors outside the grid are ignored).
        """
        neighbors = []
        if self.neighborhood_type == 'von_neumann':
            deltas = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # N, S, E, W
        else:  # moore
            deltas = [(dr, dc) for dr in [-1, 0, 1] for dc in [-1, 0, 1] if not (dr == 0 and dc == 0)]

        for dr, dc in deltas:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                neighbors.append((nr, nc))
        return neighbors

    def step(self):
        """
        Advance the simulation by one time step.
        A new grid is computed based on the current grid and the rules.
        """
        new_grid = np.copy(self.grid)

        for r in range(self.size):
            for c in range(self.size):
                current_state = self.grid[r, c]

                if current_state == EMPTY:
                    # Rule 1 (for empty areas): Chance of new tree growth
                    if random.random() < self.p:
                        new_grid[r, c] = TREE
                    # Else, remains EMPTY (already set in new_grid copy)

                elif current_state == TREE:
                    # Check for burning neighbors (Rule 2)
                    is_neighbor_burning = False
                    for nr, nc in self._get_neighbors(r, c):
                        if self.grid[nr, nc] == BURNING:
                            is_neighbor_burning = True
                            break
                    
                    if is_neighbor_burning:
                        new_grid[r, c] = BURNING
                    else:
                        # Rule 3: Chance of spontaneous ignition
                        if random.random() < self.f:
                            new_grid[r, c] = BURNING
                        # Else, remains TREE (already set in new_grid copy)
                
                elif current_state == BURNING:
                    # Rule 4: Burning tree burns out
                    new_grid[r, c] = BURNT
                
                elif current_state == BURNT:
                    # Rule 1 (for burnt trees): Chance of new tree growth or becomes empty
                    if random.random() < self.p:
                        new_grid[r, c] = TREE
                    else:
                        new_grid[r, c] = EMPTY # "remains empty" if not regrown

        self.grid = new_grid
        return self.grid


# --- Visualization Setup ---

# Create the simulation instance
forest_fire_sim = ForestFireAutomaton(
    size=GRID_SIZE,
    p_regrowth=P_REGROWTH,
    f_ignition=F_IGNITION,
    initial_density=INITIAL_FOREST_DENSITY,
    neighborhood_type=NEIGHBORHOOD_TYPE
)

# Set up the figure and axis for plotting
fig, ax = plt.subplots(figsize=(8, 8)) # Example images are square
fig.canvas.manager.set_window_title('Forest Fire Simulation') # Set window title
ax.set_title('Forest Fire') # Set plot title

# Define the colormap for visualization:
# EMPTY: Black
# TREE: Green
# BURNING: OrangeRed
# BURNT: Black (visually same as EMPTY, as per problem images)
# The states are 0 (EMPTY), 1 (TREE), 2 (BURNING), 3 (BURNT)
colors = ['black', 'forestgreen', 'orangered', 'black'] 
# Using 'forestgreen' for a nice tree color. 'orangered' for fire.
# State 3 (BURNT) is mapped to black, same as State 0 (EMPTY).
cmap = ListedColormap(colors)

# Initialize the image display using imshow
# `origin='lower'` makes (0,0) at the bottom-left, matching typical Cartesian coordinates
# and the example image's y-axis.
# `extent` defines the data coordinates of the image.
img = ax.imshow(forest_fire_sim.grid, cmap=cmap, vmin=EMPTY, vmax=BURNT,
                origin='lower', extent=[0, GRID_SIZE, 0, GRID_SIZE])

# Set axis properties to match the example images
ax.set_xticks(np.arange(0, GRID_SIZE + 1, 20))
ax.set_yticks(np.arange(0, GRID_SIZE + 1, 20))
ax.set_xlim(0, GRID_SIZE)
ax.set_ylim(0, GRID_SIZE)

# Animation update function
def update_frame(frame_num):
    """
    Called for each frame of the animation.
    It advances the simulation by one step and updates the displayed image.
    """
    new_grid_data = forest_fire_sim.step()
    img.set_data(new_grid_data)
    return [img] # Return a list of artists that have been modified

# Create and run the animationx
# `interval` is the delay between frames in milliseconds.
# `blit=True` optimizes drawing by only redrawing what has changed.
# `frames=None` (or omitted) means the animation runs indefinitely until the window is closed.
ani = animation.FuncAnimation(fig, update_frame, interval=1000 / 60, blit=True) # -> 60 fps

# Display the plot
plt.show()