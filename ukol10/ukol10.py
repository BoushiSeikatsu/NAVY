import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import random
import multiprocessing

# --- Configuration Parameters & Reproducibility ---
RANDOM_SEED = 42
A_MIN = 0.0  # Min 'a' for logistic map
A_MAX = 4.0  # Max 'a' for logistic map
INITIAL_X = 0.2 # Initial x value for simulations and data generation

# For generating (x_n, x_{n+1}) pairs for training the iterative predictor:
N_A_VALUES_TRAINING = 600    # Number of 'a' values to sample across [A_MIN, A_MAX] for training data
N_ITER_PER_A_TRAINING = 1000 # Number of (x_n, x_{n+1}) transition pairs collected for each 'a'
N_WARMUP_ITER_TRAINING = 300 # Warmup iterations for each 'a' before collecting training pairs

# For NN iterative prediction (simulation of bifurcation diagram)
N_A_VALUES_SIMULATION = 400    # Number of 'a' points for the NN-driven bifurcation diagram
N_TRANSIENTS_SIMULATION = 800  # Transients to discard during NN simulation
N_COLLECT_SIMULATION = 600     # Points to collect for each 'a' during NN simulation

# Neural Network (PyTorch for iterative predictor)
NN_EPOCHS = 40                 # Number of training epochs (adjust based on convergence)
NN_BATCH_SIZE = 1024           # Batch size for training
VALIDATION_SPLIT = 0.15        # Proportion of training data for validation
LEARNING_RATE = 0.0005         # Learning rate for the Adam optimizer
HIDDEN_LAYER_SIZES_ITERATIVE = [128, 256, 256, 128] # Hidden layer architecture for x_{n+1} = f(a, x_n)
NUM_DATALOADER_WORKERS = 0     # Set to 0 for Windows or if multiprocessing issues arise

# Set Seeds and Device
def set_seeds(seed_value):
    """Sets random seeds for numpy, torch, and python for reproducibility."""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # if using multiple GPUs
    random.seed(seed_value)
    print(f"Random seeds set to: {seed_value}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_pin_memory = True if device.type == 'cuda' else False


# Logistic Map Function (Ground Truth)
def logistic_map_func(x_n, a):
    """Calculates the next state of the logistic map: x_{n+1} = a * x_n * (1 - x_n)."""
    return a * x_n * (1 - x_n)

# Function to Generate Ground Truth Bifurcation Data (for comparison)
def generate_ground_truth_bifurcation_data(a_min, a_max, num_a_values, x0, num_transients, num_collect):
    """Generates data points (a, x) for the true bifurcation diagram."""
    print("Generating ground truth bifurcation data...")
    a_params = np.linspace(a_min, a_max, num_a_values)
    all_a_points = np.zeros(num_a_values * num_collect)
    all_x_points = np.zeros(num_a_values * num_collect)
    current_idx = 0
    for i, a_val in enumerate(a_params):
        x = x0
        for _ in range(num_transients):
            x = logistic_map_func(x, a_val)
        for _ in range(num_collect):
            x = logistic_map_func(x, a_val)
            all_a_points[current_idx] = a_val
            all_x_points[current_idx] = x
            current_idx += 1
        if (i + 1) % max(1, num_a_values // 20) == 0:
            print(f"  GT Bifurcation: Processed {i+1}/{num_a_values} 'a' values...")
    print("Ground truth bifurcation data generation complete.")
    return all_a_points, all_x_points


# Generate Training Data for Iterative Predictor: (a, x_n) -> x_{n+1}
def generate_iterative_training_data(a_min, a_max, num_a_values, x0, num_warmup, num_collect_pairs):
    """Generates training data where input is [a, x_n] and target is x_{n+1}."""
    print("Generating training data for iterative NN: (a, x_n) -> x_{n+1}...")
    a_params_train = np.linspace(a_min, a_max, num_a_values)
    
    input_features_list = [] # Stores [a, x_n]
    target_outputs_list = [] # Stores x_{n+1}

    for i, a_val_train in enumerate(a_params_train):
        x_current_train = x0
        # Warmup iterations for this 'a'
        for _ in range(num_warmup):
            x_current_train = logistic_map_func(x_current_train, a_val_train)
        
        # Collect (x_n, x_{n+1}) pairs
        for _ in range(num_collect_pairs):
            x_next_train = logistic_map_func(x_current_train, a_val_train)
            input_features_list.append([a_val_train, x_current_train])
            target_outputs_list.append(x_next_train)
            x_current_train = x_next_train # Update x_current for the next pair
            
        if (i + 1) % max(1, num_a_values // 20) == 0:
            print(f"  Iterative Training Data: Processed {i+1}/{num_a_values} 'a' values...")
            
    print("Iterative training data generation complete.")
    return np.array(input_features_list, dtype=np.float32), \
           np.array(target_outputs_list, dtype=np.float32).reshape(-1, 1)

# Iterative Predictor Neural Network Definition
class IterativeLogisticPredictor(nn.Module):
    """Neural network to predict x_{n+1} given (a, x_n)."""
    def __init__(self, input_size=2, output_size=1, hidden_sizes=None):
        super(IterativeLogisticPredictor, self).__init__()
        if hidden_sizes is None: hidden_sizes = [128, 256, 128] # Default
        
        layers = []
        prev_size = input_size
        for h_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(0.1)) # Optional: consider adding dropout
            prev_size = h_size
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Sigmoid()) # Output x_{n+1} is bounded [0,1]
        self.network = nn.Sequential(*layers)

    def forward(self, x_input): # x_input is expected to be a tensor of shape [batch_size, 2]
        return self.network(x_input)

# Training Function for the Neural Network
def train_nn_model_pytorch(model, train_loader, val_loader, num_epochs, learning_rate, dev):
    """Trains the provided PyTorch model."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True) # Optional
    
    history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
    print(f"\nStarting NN training for {num_epochs} epochs on {dev}...")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss, epoch_train_mae = 0.0, 0.0
        for inputs_batch, targets_batch in train_loader:
            inputs_batch, targets_batch = inputs_batch.to(dev), targets_batch.to(dev)
            optimizer.zero_grad()
            outputs_batch = model(inputs_batch)
            loss = criterion(outputs_batch, targets_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * inputs_batch.size(0)
            epoch_train_mae += torch.abs(outputs_batch - targets_batch).sum().item()
        
        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        avg_train_mae = epoch_train_mae / len(train_loader.dataset)
        history['loss'].append(avg_train_loss)
        history['mae'].append(avg_train_mae)

        model.eval()
        epoch_val_loss, epoch_val_mae = 0.0, 0.0
        with torch.no_grad():
            for inputs_batch, targets_batch in val_loader:
                inputs_batch, targets_batch = inputs_batch.to(dev), targets_batch.to(dev)
                outputs_batch = model(inputs_batch)
                loss = criterion(outputs_batch, targets_batch)
                epoch_val_loss += loss.item() * inputs_batch.size(0)
                epoch_val_mae += torch.abs(outputs_batch - targets_batch).sum().item()
        
        avg_val_loss = epoch_val_loss / len(val_loader.dataset)
        avg_val_mae = epoch_val_mae / len(val_loader.dataset)
        history['val_loss'].append(avg_val_loss)
        history['val_mae'].append(avg_val_mae)
        
        # if scheduler: scheduler.step(avg_val_loss) # For ReduceLROnPlateau

        print(f"Epoch [{epoch+1:03d}/{num_epochs:03d}] | "
              f"Train Loss: {avg_train_loss:.6f}, Train MAE: {avg_train_mae:.6f} | "
              f"Val Loss: {avg_val_loss:.6f}, Val MAE: {avg_val_mae:.6f}")
              # f"LR: {optimizer.param_groups[0]['lr']:.1e}") # If scheduler used

    print("NN training complete.")
    return model, history

# Simulate Bifurcation using the Trained Iterative NN
def simulate_bifurcation_with_nn(model, a_min_sim, a_max_sim, num_a_vals_sim, 
                                 x0_sim, num_trans_sim, num_collect_sim, dev):
    """Simulates the bifurcation diagram by iteratively applying the trained NN."""
    print("\nSimulating bifurcation diagram using the trained iterative NN...")
    a_params_for_sim = np.linspace(a_min_sim, a_max_sim, num_a_vals_sim)
    
    all_a_nn_sim_list = []
    all_x_nn_sim_list = []
    
    model.eval() # Ensure model is in evaluation mode
    with torch.no_grad(): # No gradients needed for inference
        for i, current_a_sim in enumerate(a_params_for_sim):
            # Initialize x for this 'a' value
            # x_predicted needs to be a 2D tensor [1,1] for model input when combined with 'a'
            x_predicted_current = torch.tensor([[x0_sim]], dtype=torch.float32).to(dev) 
            # 'a' value also needs to be a 2D tensor [1,1]
            a_val_tensor_sim = torch.tensor([[current_a_sim]], dtype=torch.float32).to(dev)

            # Discard transients using the NN
            for _ in range(num_trans_sim):
                # Prepare model input: concatenate 'a' and current 'x'
                model_input_tensor = torch.cat((a_val_tensor_sim, x_predicted_current), dim=1) # Shape: [1, 2]
                x_predicted_current = model(model_input_tensor) # NN predicts x_{n+1}
            
            # Collect points for the diagram using the NN
            for _ in range(num_collect_sim):
                model_input_tensor = torch.cat((a_val_tensor_sim, x_predicted_current), dim=1)
                x_predicted_current = model(model_input_tensor)
                all_a_nn_sim_list.append(current_a_sim)
                all_x_nn_sim_list.append(x_predicted_current.item()) # .item() converts scalar tensor to Python number
            
            if (i + 1) % max(1, num_a_vals_sim // 20) == 0:
                print(f"  NN Simulation: Processed {i+1}/{num_a_vals_sim} 'a' values...")
    
    print("NN-driven bifurcation simulation complete.")
    return np.array(all_a_nn_sim_list), np.array(all_x_nn_sim_list)


# --- 6. Main Execution Block ---
def main():
    """Main function to run the complete workflow."""
    print(f"Using PyTorch device: {device}")
    print(f"Pin memory for DataLoader: {use_pin_memory}, Num workers: {NUM_DATALOADER_WORKERS}")

    set_seeds(RANDOM_SEED)
    plt.style.use('seaborn-v0_8-whitegrid') # Apply a style for plots

    # Generate Ground Truth Bifurcation Data (for visual comparison)
    # These parameters should match the NN simulation parameters for a fair visual comparison.
    gt_a_points, gt_x_points = generate_ground_truth_bifurcation_data(
        A_MIN, A_MAX, N_A_VALUES_SIMULATION, 
        INITIAL_X, N_TRANSIENTS_SIMULATION, N_COLLECT_SIMULATION
    )

    # Generate Training Data for the Iterative Predictor NN
    # This data is of the form: input=[a, x_n], target=x_{n+1}
    X_train_iter_np, y_train_iter_np = generate_iterative_training_data(
        A_MIN, A_MAX, N_A_VALUES_TRAINING, 
        INITIAL_X, N_WARMUP_ITER_TRAINING, N_ITER_PER_A_TRAINING
    )
    
    # Convert to PyTorch Tensors and create DataLoaders
    X_tensor_iter = torch.tensor(X_train_iter_np, dtype=torch.float32) # Shape: [N_samples, 2]
    y_tensor_iter = torch.tensor(y_train_iter_np, dtype=torch.float32) # Shape: [N_samples, 1]
    
    full_dataset_iter = TensorDataset(X_tensor_iter, y_tensor_iter)
    dataset_size_iter = len(full_dataset_iter)
    val_size_iter = int(np.floor(VALIDATION_SPLIT * dataset_size_iter))
    train_size_iter = dataset_size_iter - val_size_iter
    
    train_dataset_iter, val_dataset_iter = random_split(
        full_dataset_iter, [train_size_iter, val_size_iter],
        generator=torch.Generator().manual_seed(RANDOM_SEED) # For reproducible split
    )
    print(f"\nIterative predictor dataset: Training samples: {len(train_dataset_iter)}, Validation samples: {len(val_dataset_iter)}")
    
    train_loader_iter = DataLoader(train_dataset_iter, batch_size=NN_BATCH_SIZE, shuffle=True, 
                                   num_workers=NUM_DATALOADER_WORKERS, pin_memory=use_pin_memory)
    val_loader_iter = DataLoader(val_dataset_iter, batch_size=NN_BATCH_SIZE, shuffle=False, 
                                 num_workers=NUM_DATALOADER_WORKERS, pin_memory=use_pin_memory)

    # Initialize and Train the Iterative Predictor NN Model
    iterative_nn_model = IterativeLogisticPredictor(
        input_size=2, output_size=1, hidden_sizes=HIDDEN_LAYER_SIZES_ITERATIVE
    ).to(device)
    print("\nIterative Predictor NN Model Architecture:"); print(iterative_nn_model)
    
    # Train the model
    iterative_nn_model, training_history = train_nn_model_pytorch(
        iterative_nn_model, train_loader_iter, val_loader_iter, 
        NN_EPOCHS, LEARNING_RATE, device
    )
    
    # Optional: Plot training history
    fig_hist, ax1_hist = plt.subplots(figsize=(12, 6))
    color = 'tab:red'; ax1_hist.set_xlabel('Epoch', fontsize=14)
    ax1_hist.set_ylabel('Loss (MSE)', color=color, fontsize=14)
    ax1_hist.plot(training_history['loss'], color=color, linestyle='-', label='Training Loss')
    ax1_hist.plot(training_history['val_loss'], color=color, linestyle='--', label='Validation Loss')
    ax1_hist.tick_params(axis='y', labelcolor=color, labelsize=11); ax1_hist.tick_params(axis='x', labelsize=11)
    ax1_hist.legend(loc='upper left', fontsize=11); ax1_hist.grid(True, linestyle=':', axis='y', alpha=0.7)
    ax2_hist = ax1_hist.twinx(); color = 'tab:blue'
    ax2_hist.set_ylabel('MAE', color=color, fontsize=14)
    ax2_hist.plot(training_history['mae'], color=color, linestyle='-', label='Training MAE')
    ax2_hist.plot(training_history['val_mae'], color=color, linestyle='--', label='Validation MAE')
    ax2_hist.tick_params(axis='y', labelcolor=color, labelsize=11)
    ax2_hist.legend(loc='upper right', fontsize=11)
    fig_hist.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
    fig_hist.suptitle('Iterative NN Training History', fontsize=16)
    plt.show()

    # Simulate Bifurcation Diagram using the Trained Iterative NN
    nn_simulated_a, nn_simulated_x = simulate_bifurcation_with_nn(
        iterative_nn_model, A_MIN, A_MAX, N_A_VALUES_SIMULATION, 
        INITIAL_X, N_TRANSIENTS_SIMULATION, N_COLLECT_SIMULATION, device
    )

    # Plot Side-by-Side Comparison
    fig_comparison, (ax_ground_truth, ax_nn_prediction) = plt.subplots(
        1, 2, figsize=(22, 9), sharey=True # Share y-axis for direct comparison
    )

    # Plot Ground Truth Bifurcation Diagram
    ax_ground_truth.plot(gt_a_points, gt_x_points, marker=',', color='black', linestyle='None', alpha=0.04)
    ax_ground_truth.set_title("Ground Truth Logistic Map Bifurcation", fontsize=17)
    ax_ground_truth.set_xlabel("Parameter 'a'", fontsize=15)
    ax_ground_truth.set_ylabel("x (Attractor Points)", fontsize=15)
    ax_ground_truth.set_xlim(A_MIN, A_MAX); ax_ground_truth.set_ylim(0, 1)
    ax_ground_truth.tick_params(axis='both', which='major', labelsize=12)
    ax_ground_truth.grid(True, linestyle='--', alpha=0.6)

    # Plot NN-Simulated Bifurcation Diagram
    ax_nn_prediction.plot(nn_simulated_a, nn_simulated_x, marker=',', color='red', linestyle='None', alpha=0.04)
    ax_nn_prediction.set_title("NN Iterative Simulation of Bifurcation", fontsize=17)
    ax_nn_prediction.set_xlabel("Parameter 'a'", fontsize=15)
    # ax_nn_prediction.set_ylabel("Predicted x", fontsize=15) # Y-label is shared
    ax_nn_prediction.set_xlim(A_MIN, A_MAX); ax_nn_prediction.set_ylim(0, 1)
    ax_nn_prediction.tick_params(axis='both', which='major', labelsize=12)
    ax_nn_prediction.grid(True, linestyle='--', alpha=0.6)
    
    fig_comparison.suptitle("Comparison: Logistic Map Dynamics vs. NN Iterative Predictor", fontsize=20, y=0.98)
    fig_comparison.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout for suptitle
    plt.show()

    print("\nFull workflow for Task 10 (Option A: Iterative NN Prediction) complete.")

# Guard for Multiprocessing (Important for Windows)
if __name__ == '__main__':
    multiprocessing.freeze_support() # Call this for PyTorch DataLoader with num_workers > 0 on Windows
    main()