import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf # Import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam # Correct import path for Adam

# GPU setup: Attempt to enable memory growth for GPUs to avoid common TensorFlow issues.
# This is good practice if a GPU is available.
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"Found {len(gpus)} Physical GPUs, configured {len(logical_gpus)} Logical GPUs with memory growth.")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(f"GPU Memory Growth Error: {e}")

# --- Configuration Parameters ---

# For logistic map generation
A_MIN = 0.0  # Minimum value of parameter 'a'
A_MAX = 4.0  # Maximum value of parameter 'a'
N_A_VALUES_DIAGRAM = 1000  # Number of 'a' points for the diagram (as suggested in problem)
INITIAL_X = 0.2          # Initial value of x (must be in (0,1), not 0 or 1)
N_TRANSIENTS = 600       # Iterations to discard to allow system to settle on the attractor
N_COLLECT = 400          # Iterations to collect for plotting/training for each 'a' (points on attractor)

# For Neural Network
NN_EPOCHS = 30           # Number of epochs for training (adjust based on convergence and time)
NN_BATCH_SIZE = 256      # Batch size for training (larger can speed up training if RAM allows)
NN_A_VALUES_PREDICT = N_A_VALUES_DIAGRAM # Number of 'a' points for plotting NN predictions

# --- 1. Implement the logistic map ---
def logistic_map_func(x_n, a):
    """
    Calculates the next state of the logistic map: x_{n+1} = a * x_n * (1 - x_n).
    
    Args:
        x_n (float): Current value of x.
        a (float): Parameter 'a' of the logistic map.
        
    Returns:
        float: Next value of x, x_{n+1}.
    """
    return a * x_n * (1 - x_n)

# --- Function to generate bifurcation data ---
def generate_bifurcation_data(a_min_val, a_max_val, num_a_values, x0, num_transients, num_collect):
    """
    Generates data points (a, x) for the bifurcation diagram.
    For each 'a', it iterates the logistic map, discards transients, and collects attractor points.
    
    Args:
        a_min_val (float): Minimum value of parameter 'a'.
        a_max_val (float): Maximum value of parameter 'a'.
        num_a_values (int): Number of 'a' values to sample in the range [a_min_val, a_max_val].
        x0 (float): Initial x value for the map iterations.
        num_transients (int): Number of initial iterations to discard.
        num_collect (int): Number of subsequent iterations to collect as attractor points.
        
    Returns:
        tuple: (numpy.ndarray, numpy.ndarray)
               - First array contains all 'a' values used.
               - Second array contains corresponding 'x' attractor points.
    """
    print("Generating bifurcation data...")
    # Create an array of 'a' parameter values using np.linspace
    a_params = np.linspace(a_min_val, a_max_val, num_a_values)
    
    # Lists to store the points for the bifurcation diagram
    all_a_points = []
    all_x_points = []

    # Iterate over each value of parameter 'a'
    for i, a_val in enumerate(a_params):
        x = x0  # Reset x to initial condition for each 'a'
        
        # Iterate for a number of steps to let the system settle (discard transients)
        for _ in range(num_transients):
            x = logistic_map_func(x, a_val)
        
        # Collect the next set of points, which should be on the attractor
        for _ in range(num_collect):
            x = logistic_map_func(x, a_val)
            all_a_points.append(a_val)
            all_x_points.append(x)
        
        # Progress indicator: Print status roughly 20 times during generation
        if (i + 1) % max(1, num_a_values // 20) == 0:
            print(f"  Processed {i+1}/{num_a_values} 'a' values for bifurcation data...")
            
    print("Bifurcation data generation complete.")
    return np.array(all_a_points), np.array(all_x_points)

# --- Function to create, train, and return a neural network model ---
def create_and_train_nn_model(X_train_data, y_train_data, num_epochs, batch_size_val):
    """
    Creates a sequential neural network, compiles it, and trains it on the provided data.
    The network is designed to predict 'x' (logistic map attractor value) given 'a'.
    
    Args:
        X_train_data (numpy.ndarray): Input data (features), expected shape (n_samples, 1).
        y_train_data (numpy.ndarray): Target data, expected shape (n_samples,).
        num_epochs (int): Number of training epochs.
        batch_size_val (int): Batch size for training.
        
    Returns:
        tuple: (tensorflow.keras.models.Sequential, tensorflow.keras.callbacks.History)
               - The trained Keras model.
               - Training history object (contains loss and metrics evolution).
    """
    print("\nCreating and training neural network...")
    
    # Define the neural network architecture: A Multi-Layer Perceptron (MLP)
    model = Sequential([
        Dense(64, activation='relu', input_shape=(1,)),  # Input layer: 1 neuron for 'a'
        Dense(128, activation='relu'),                   # Hidden layer 1
        Dense(128, activation='relu'),                   # Hidden layer 2
        Dense(64, activation='relu'),                    # Hidden layer 3
        Dense(1, activation='sigmoid')                   # Output layer: 1 neuron for 'x', sigmoid for [0,1] range
    ])

    # Compile the model
    # Adam optimizer is a common and effective choice. 
    # Mean Squared Error (MSE) is suitable for this regression task.
    # Mean Absolute Error (MAE) is included as an additional metric.
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='mean_squared_error',
                  metrics=['mae']) 

    # Print a summary of the model architecture
    model.summary()

    # Train the model
    # A portion of the training data (10%) is used for validation to monitor overfitting.
    print(f"Starting training for {num_epochs} epochs with batch size {batch_size_val}...")
    history = model.fit(X_train_data, y_train_data,
                        epochs=num_epochs,
                        batch_size=batch_size_val,
                        validation_split=0.1, # Use 10% of data for validation
                        verbose=1) # Show progress bar during training (0=silent, 1=bar, 2=one line per epoch)
                        
    print("Neural network training complete.")
    return model, history


# --- Main execution block ---
# This script is designed to be run from top to bottom.
# The problem states "You dont need to do unit tests nor main function parameters."

# --- Step 1: Implement the logistic map ---
# The `logistic_map_func` is defined above and will be used by `generate_bifurcation_data`.

# --- Step 2: Visualize the bifurcation diagram for different values of parameter a ---
# Generate the data points for the bifurcation diagram.
# These points (a_diagram_data, x_diagram_data) will also serve as the training data for the NN.
a_diagram_data, x_diagram_data = generate_bifurcation_data(
    A_MIN, A_MAX, N_A_VALUES_DIAGRAM, INITIAL_X, N_TRANSIENTS, N_COLLECT
)

# Plot the original bifurcation diagram
plt.style.use('seaborn-v0_8-whitegrid') # Using a seaborn style for aesthetically pleasing plots
plt.figure(figsize=(14, 8))
# Using small, semi-transparent black dots (',k') for a dense plot. 'k' means black.
# alpha controls transparency; lower values are more transparent.
plt.plot(a_diagram_data, x_diagram_data, ',k', alpha=0.05) 
plt.title("Bifurcation Diagram of the Logistic Map (Generated Data)", fontsize=18, pad=15)
plt.xlabel("Parameter 'a'", fontsize=15)
plt.ylabel("x (Attractor Points)", fontsize=15)
plt.xlim(A_MIN, A_MAX) # Set x-axis limits
plt.ylim(0, 1)         # Set y-axis limits (x values are between 0 and 1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# --- Step 3: Use neural network to predict the values of the logistic map attractor ---
# The problem statement says "predict the number of the logistic map".
# This is interpreted as predicting the 'x' values (attractor points) for a given 'a'.
# The neural network will learn a function f(a) = x_predicted.
# Since the attractor can have multiple x values for a single 'a' (e.g., in 2-cycles or chaos),
# and the network has a single output neuron trained with MSE, it will tend to predict
# the conditional mean of x given a, E[x|a]. This means it will predict a sort of "average"
# path through the bifurcation diagram, similar to "Example 3" in the problem description.

# Prepare data for the neural network:
# X_train should be 'a' values, reshaped to (n_samples, 1) as Keras expects.
# y_train should be corresponding 'x' values, shape (n_samples,).
X_train_nn = a_diagram_data.reshape(-1, 1)
y_train_nn = x_diagram_data

# Create and train the neural network model
# Note: Training duration depends on data size (N_A_VALUES_DIAGRAM * N_COLLECT) and NN_EPOCHS.
# With current settings (1000 * 400 = 400,000 samples), 30 epochs might take a few minutes.
nn_model, training_history = create_and_train_nn_model(
    X_train_nn, y_train_nn, NN_EPOCHS, NN_BATCH_SIZE
)

# Plot training & validation loss and MAE to assess model performance during training.
# This helps to check for overfitting or if more training is needed.
fig, ax1 = plt.subplots(figsize=(12, 7))

color = 'tab:red'
ax1.set_xlabel('Epoch', fontsize=15)
ax1.set_ylabel('Loss (MSE)', color=color, fontsize=15)
ax1.plot(training_history.history['loss'], color=color, linestyle='-', label='Training Loss (MSE)')
ax1.plot(training_history.history['val_loss'], color=color, linestyle='--', label='Validation Loss (MSE)')
ax1.tick_params(axis='y', labelcolor=color, labelsize=12)
ax1.tick_params(axis='x', labelsize=12)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()  # Instantiate a second y-axis that shares the same x-axis for MAE
color = 'tab:blue'
ax2.set_ylabel('Mean Absolute Error (MAE)', color=color, fontsize=15) # MAE y-axis
ax2.plot(training_history.history['mae'], color=color, linestyle='-', label='Training MAE')
ax2.plot(training_history.history['val_mae'], color=color, linestyle='--', label='Validation MAE')
ax2.tick_params(axis='y', labelcolor=color, labelsize=12)
ax2.legend(loc='upper right')

fig.tight_layout() # Adjust plot to prevent labels from being clipped
plt.title('Model Training History (Loss and MAE)', fontsize=18, pad=15)
plt.show()

# --- Step 4: Visualize the bifurcation map predicted by the neural network ---
# Generate a range of 'a' values for making predictions with the trained NN.
# These can be the same as, or denser than, the 'a' values used for training.
a_values_for_nn_prediction = np.linspace(A_MIN, A_MAX, NN_A_VALUES_PREDICT).reshape(-1, 1)

# Make predictions using the trained neural network
print("\nMaking predictions with the trained neural network...")
x_predicted_by_nn = nn_model.predict(a_values_for_nn_prediction)
print("Predictions complete.")

# Plot the original bifurcation diagram with the neural network's predictions overlaid.
# This visualization corresponds to "Example 3" in the problem description.
plt.figure(figsize=(14, 8))
# Plot original data points (ground truth)
plt.plot(a_diagram_data, x_diagram_data, ',k', alpha=0.05, label='Actual Attractor Points')
# Plot NN predictions
# a_values_for_nn_prediction is shape (N,1), x_predicted_by_nn is shape (N,1)
# .flatten() converts them to 1D arrays suitable for plt.plot
plt.plot(a_values_for_nn_prediction.flatten(), x_predicted_by_nn.flatten(), 
         '.r', markersize=2.5, label='NN Prediction (Mean Behavior)') # Red dots for prediction
plt.title("Bifurcation Diagram with Neural Network Predictions", fontsize=18, pad=15)
plt.xlabel("Parameter 'a'", fontsize=15)
plt.ylabel("x (Population / Attractor Value)", fontsize=15)
plt.xlim(A_MIN, A_MAX)
plt.ylim(0, 1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# Increase marker size in legend for better visibility of the prediction points
legend = plt.legend(markerscale=4) 
# Ensure legend markers for scatter plots are opaque if original markers are transparent
for lh in legend.legendHandles: 
    try: # Check if it's a Line2D object (used for scatter in legend)
        if hasattr(lh, 'set_alpha'): lh.set_alpha(1)
    except AttributeError:
        pass # Handle other types of legend handles if necessary

plt.show()

print("\nTask 10 solution execution complete. All plots have been generated.")