import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import random # For setting Python's random seed
import multiprocessing # For freeze_support

# --- Configuration Parameters & Reproducibility ---
# General settings
RANDOM_SEED = 42 # For consistent results

# Logistic map generation
A_MIN = 0.0
A_MAX = 4.0
N_A_VALUES_DIAGRAM = 1000
INITIAL_X = 0.2
N_TRANSIENTS = 1000
N_COLLECT = 500

# Neural Network (PyTorch)
NN_EPOCHS = 20 # Reduced for quicker testing, you can increase it back
NN_BATCH_SIZE = 256
NN_A_VALUES_PREDICT = N_A_VALUES_DIAGRAM
VALIDATION_SPLIT = 0.15
LEARNING_RATE = 0.0005
HIDDEN_LAYER_SIZES = [128, 256, 128]
NUM_DATALOADER_WORKERS = 2 # Set to 0 if issues persist or for simpler debugging

# --- Set Seeds for Reproducibility ---
def set_seeds(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    random.seed(seed_value)
    print(f"Random seeds set to: {seed_value}")

# --- PyTorch Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Determine if pin_memory should be used
use_pin_memory = True if device.type == 'cuda' else False


# --- 1. Implement the logistic map ---
def logistic_map_func(x_n, a):
    return a * x_n * (1 - x_n)

# --- Function to generate bifurcation data ---
def generate_bifurcation_data(a_min_val, a_max_val, num_a_values, x0, num_transients, num_collect):
    print("Generating bifurcation data...")
    a_params = np.linspace(a_min_val, a_max_val, num_a_values)
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
            print(f"  Processed {i+1}/{num_a_values} 'a' values for bifurcation data...")
    print("Bifurcation data generation complete.")
    return all_a_points, all_x_points

# --- PyTorch Neural Network Definition ---
class LogisticMapPredictor(nn.Module):
    def __init__(self, input_size=1, output_size=1, hidden_sizes=None):
        super(LogisticMapPredictor, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 128, 64]
        layers = []
        prev_size = input_size
        for h_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.ReLU())
            prev_size = h_size
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# --- Function to train the PyTorch neural network model ---
def train_nn_model_pytorch(model, train_loader, val_loader, num_epochs, learning_rate, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
    print(f"\nStarting training for {num_epochs} epochs on {device}...")
    print(f"Optimizer: Adam, Learning Rate: {learning_rate}")
    print(f"Loss Function: MSE")

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_mae = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * inputs.size(0)
            epoch_train_mae += torch.abs(outputs - targets).sum().item()
        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        avg_train_mae = epoch_train_mae / len(train_loader.dataset)
        history['loss'].append(avg_train_loss)
        history['mae'].append(avg_train_mae)

        model.eval()
        epoch_val_loss = 0.0
        epoch_val_mae = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item() * inputs.size(0)
                epoch_val_mae += torch.abs(outputs - targets).sum().item()
        avg_val_loss = epoch_val_loss / len(val_loader.dataset)
        avg_val_mae = epoch_val_mae / len(val_loader.dataset)
        history['val_loss'].append(avg_val_loss)
        history['val_mae'].append(avg_val_mae)
        print(f"Epoch [{epoch+1:03d}/{num_epochs:03d}] | "
              f"Train Loss: {avg_train_loss:.6f} | Train MAE: {avg_train_mae:.6f} | "
              f"Val Loss: {avg_val_loss:.6f} | Val MAE: {avg_val_mae:.6f}")
    print("PyTorch neural network training complete.")
    return model, history

# --- Main execution function ---
def main():
    print(f"Using PyTorch device: {device}")
    print(f"Pin memory for DataLoader: {use_pin_memory}")
    print(f"Number of DataLoader workers: {NUM_DATALOADER_WORKERS}")

    set_seeds(RANDOM_SEED)

    # --- Step 1: Implement the logistic map (done via logistic_map_func) ---

    # --- Step 2: Visualize the bifurcation diagram ---
    a_diagram_data, x_diagram_data = generate_bifurcation_data(
        A_MIN, A_MAX, N_A_VALUES_DIAGRAM, INITIAL_X, N_TRANSIENTS, N_COLLECT
    )
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 8))
    plt.plot(a_diagram_data, x_diagram_data, marker=',', color='k', linestyle='None', alpha=0.03)
    plt.title("Bifurcation Diagram of the Logistic Map (Generated Ground Truth)", fontsize=18, pad=15)
    plt.xlabel("Parameter 'a'", fontsize=15)
    plt.ylabel("x (Attractor Points)", fontsize=15)
    plt.xlim(A_MIN, A_MAX)
    plt.ylim(0, 1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # --- Step 3: Use neural network (PyTorch) to predict ---
    X_tensor_nn = torch.tensor(a_diagram_data.reshape(-1, 1), dtype=torch.float32)
    y_tensor_nn = torch.tensor(x_diagram_data.reshape(-1, 1), dtype=torch.float32)
    full_dataset = TensorDataset(X_tensor_nn, y_tensor_nn)
    dataset_size = len(full_dataset)
    val_size = int(np.floor(VALIDATION_SPLIT * dataset_size))
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(RANDOM_SEED))
    print(f"\nDataset split: Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=NN_BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_DATALOADER_WORKERS, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=NN_BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_DATALOADER_WORKERS, pin_memory=use_pin_memory)
    
    nn_model_pytorch = LogisticMapPredictor(hidden_sizes=HIDDEN_LAYER_SIZES).to(device)
    print("\nModel Architecture:")
    print(nn_model_pytorch)
    nn_model_pytorch, training_history_pytorch = train_nn_model_pytorch(
        nn_model_pytorch, train_loader, val_loader, NN_EPOCHS, LEARNING_RATE, device
    )

    fig, ax1 = plt.subplots(figsize=(12, 7))
    color = 'tab:red'
    ax1.set_xlabel('Epoch', fontsize=15)
    ax1.set_ylabel('Loss (MSE)', color=color, fontsize=15)
    ax1.plot(training_history_pytorch['loss'], color=color, linestyle='-', marker='.', markersize=5, label='Training Loss (MSE)')
    ax1.plot(training_history_pytorch['val_loss'], color=color, linestyle='--', marker='x', markersize=5, label='Validation Loss (MSE)')
    ax1.tick_params(axis='y', labelcolor=color, labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.legend(loc='upper left', fontsize=12)
    ax1.grid(True, linestyle=':', axis='y')
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Mean Absolute Error (MAE)', color=color, fontsize=15)
    ax2.plot(training_history_pytorch['mae'], color=color, linestyle='-', marker='.', markersize=5, label='Training MAE')
    ax2.plot(training_history_pytorch['val_mae'], color=color, linestyle='--', marker='x', markersize=5, label='Validation MAE')
    ax2.tick_params(axis='y', labelcolor=color, labelsize=12)
    ax2.legend(loc='upper right', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.title('PyTorch Model Training History (Loss and MAE)', fontsize=18, pad=20)
    plt.show()

    # --- Step 4: Visualize the bifurcation map predicted ---
    a_values_for_nn_prediction_np = np.linspace(A_MIN, A_MAX, NN_A_VALUES_PREDICT).reshape(-1, 1)
    a_tensor_for_prediction = torch.tensor(a_values_for_nn_prediction_np, dtype=torch.float32).to(device)
    print("\nMaking predictions with the trained PyTorch neural network...")
    nn_model_pytorch.eval()
    with torch.no_grad():
        x_predicted_by_nn_tensor = nn_model_pytorch(a_tensor_for_prediction)
    x_predicted_by_nn_pytorch = x_predicted_by_nn_tensor.cpu().numpy()
    print("Predictions complete.")

    plt.figure(figsize=(16, 9))
    plt.plot(a_diagram_data, x_diagram_data, marker=',', color='k', linestyle='None', alpha=0.03, label='Actual Attractor Points')
    plt.plot(a_values_for_nn_prediction_np.flatten(), x_predicted_by_nn_pytorch.flatten(),
             color='red', linestyle='-', linewidth=2,
             label=f'NN Prediction (Mean Behavior - PyTorch)')
    plt.title("Bifurcation Diagram with PyTorch Neural Network Predictions", fontsize=20, pad=15)
    plt.xlabel("Parameter 'a'", fontsize=16)
    plt.ylabel("x (Population / Attractor Value)", fontsize=16)
    plt.xlim(A_MIN, A_MAX)
    plt.ylim(0, 1)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    legend = plt.legend(fontsize=12, markerscale=10)
    for lh in legend.legend_handles: 
        if hasattr(lh, 'set_alpha'): lh.set_alpha(1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    print("All plots have been generated and displayed.")

# --- Guard for multiprocessing ---
if __name__ == '__main__':
    multiprocessing.freeze_support() # Necessary for Windows when using num_workers > 0
    main()
