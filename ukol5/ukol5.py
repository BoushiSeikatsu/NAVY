import gymnasium as gym
import numpy as np
# No PyTorch or TensorFlow imports!
from collections import deque
import random
import imageio # For creating GIFs
import matplotlib.pyplot as plt # For plotting training progress
import math # For sqrt, etc.

# --- Hyperparameters ---
STATE_SIZE = 4
ACTION_SIZE = 2

GAMMA = 0.99
LEARNING_RATE = 0.001
MEMORY_SIZE = 10000
BATCH_SIZE = 64
HIDDEN_SIZE = 32

EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_STEPS = 30000 # Might need more steps for custom implementation
TARGET_UPDATE_FREQ = 100

NUM_EPISODES_TRAIN = 400 # Increased, as custom might learn slower
MAX_STEPS_PER_EPISODE = 500

NUM_EPISODES_TEST = 5

# --- Custom Neural Network Components ---

def xavier_uniform_init(fan_in, fan_out):
    """Xavier/Glorot uniform initialization for weights."""
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=(fan_in, fan_out))

class CustomLinearLayer:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Initialize weights and biases
        self.weights = xavier_uniform_init(input_dim, output_dim)
        self.biases = np.zeros(output_dim)
        
        self.input_data = None # Store for backpropagation
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_biases = np.zeros_like(self.biases)

    def forward(self, input_data):
        self.input_data = input_data
        # Output = Input @ Weights + Biases
        return np.dot(input_data, self.weights) + self.biases

    def backward(self, grad_output):
        # grad_output shape: (batch_size, output_dim)
        
        # Gradient w.r.t. weights: dL/dW = X.T @ dL/dO
        self.grad_weights = np.dot(self.input_data.T, grad_output)
        
        # Gradient w.r.t. biases: dL/dB = sum(dL/dO, axis=0)
        self.grad_biases = np.sum(grad_output, axis=0)
        
        # Gradient w.r.t. input to this layer: dL/dX = dL/dO @ W.T
        grad_input = np.dot(grad_output, self.weights.T)
        
        return grad_input

class CustomReLU:
    def __init__(self):
        self.input_data = None # Store for backpropagation

    def forward(self, input_data):
        self.input_data = input_data
        return np.maximum(0, input_data)

    def backward(self, grad_output):
        # Derivative of ReLU is 1 if input > 0, else 0
        relu_deriv = (self.input_data > 0).astype(float)
        return grad_output * relu_deriv

class CustomQNetwork:
    def __init__(self, state_size, action_size, hidden_size):
        self.fc1 = CustomLinearLayer(state_size, hidden_size)
        self.relu1 = CustomReLU()
        self.fc2 = CustomLinearLayer(hidden_size, hidden_size)
        self.relu2 = CustomReLU()
        self.fc3 = CustomLinearLayer(hidden_size, action_size)

        # List of layers with parameters for optimizer
        self.parameterized_layers = [self.fc1, self.fc2, self.fc3]

    def forward(self, state):
        x = self.fc1.forward(state)
        x = self.relu1.forward(x)
        x = self.fc2.forward(x)
        x = self.relu2.forward(x)
        x = self.fc3.forward(x)
        return x

    def backward(self, grad_output):
        # Propagate gradients backward through the network
        grad_output = self.fc3.backward(grad_output)
        grad_output = self.relu2.backward(grad_output)
        grad_output = self.fc2.backward(grad_output)
        grad_output = self.relu1.backward(grad_output)
        grad_output = self.fc1.backward(grad_output)
        # Gradients are now stored in each layer's .grad_weights and .grad_biases

    def get_weights_list(self):
        """Returns a list of (weights, biases) tuples for each linear layer, deep copied."""
        return [
            (np.copy(layer.weights), np.copy(layer.biases))
            for layer in self.parameterized_layers
        ]

    def set_weights_list(self, weights_list):
        """Sets weights and biases for each linear layer from a list."""
        for i, layer in enumerate(self.parameterized_layers):
            layer.weights, layer.biases = np.copy(weights_list[i][0]), np.copy(weights_list[i][1])

class CustomAdamOptimizer:
    def __init__(self, parameterized_layers, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.layers = parameterized_layers # List of CustomLinearLayer instances
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0 # Timestep

        self.m_weights, self.v_weights = [], []
        self.m_biases, self.v_biases = [], []

        for layer in self.layers:
            self.m_weights.append(np.zeros_like(layer.weights))
            self.v_weights.append(np.zeros_like(layer.weights))
            self.m_biases.append(np.zeros_like(layer.biases))
            self.v_biases.append(np.zeros_like(layer.biases))
                
    def step(self):
        self.t += 1
        for i, layer in enumerate(self.layers):
            # Update weights
            grad_w = layer.grad_weights
            self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * grad_w
            self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * (grad_w ** 2)
            
            m_hat_w = self.m_weights[i] / (1 - self.beta1 ** self.t)
            v_hat_w = self.v_weights[i] / (1 - self.beta2 ** self.t)
            
            layer.weights -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)

            # Update biases
            grad_b = layer.grad_biases
            self.m_biases[i] = self.beta1 * self.m_biases[i] + (1 - self.beta1) * grad_b
            self.v_biases[i] = self.beta2 * self.v_biases[i] + (1 - self.beta2) * (grad_b ** 2)

            m_hat_b = self.m_biases[i] / (1 - self.beta1 ** self.t)
            v_hat_b = self.v_biases[i] / (1 - self.beta2 ** self.t)

            layer.biases -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

# --- Replay Buffer (remains the same) ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # Ensure conversion to numpy arrays here
        return np.array(states, dtype=np.float32), \
               np.array(actions, dtype=np.int64), \
               np.array(rewards, dtype=np.float32), \
               np.array(next_states, dtype=np.float32), \
               np.array(dones, dtype=np.bool_)


    def __len__(self):
        return len(self.buffer)

# --- DQN Agent (Custom Implementation) ---
class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size):
        self.state_size = state_size
        self.action_size = action_size

        self.replay_buffer = ReplayBuffer(MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.epsilon_decay_value = (EPSILON_START - EPSILON_END) / EPSILON_DECAY_STEPS
        self.total_steps = 0

        self.q_network = CustomQNetwork(state_size, action_size, hidden_size)
        self.target_network = CustomQNetwork(state_size, action_size, hidden_size)
        
        self.optimizer = CustomAdamOptimizer(
            self.q_network.parameterized_layers, # Pass layers with parameters
            learning_rate=LEARNING_RATE
        )
        
        self.update_target_network()

    def update_target_network(self):
        weights_list = self.q_network.get_weights_list()
        self.target_network.set_weights_list(weights_list)
        # print("Target network updated.") # Can be noisy

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def act(self, state, training=True): # state is a 1D numpy array
        self.total_steps += 1
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_batch = np.reshape(state, [1, self.state_size]) # Network expects batch
        q_values = self.q_network.forward(state_batch)
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)

        # 1. Get Q-values for next states from Target Network: max_a' Q_target(s', a')
        next_q_values_target_net = self.target_network.forward(next_states) # (B, A)
        max_next_q_values = np.max(next_q_values_target_net, axis=1) # (B,)

        # 2. Calculate TD targets: R + gamma * max_a' Q_target(s', a') * (1-done)
        td_targets_for_actions_taken = rewards + GAMMA * max_next_q_values * (~dones) # ~dones is (1-dones) for bool

        # 3. Get Q-values for current states from Q-Network: Q_online(s,a) for all 'a'
        current_q_values_all_actions = self.q_network.forward(states) # (B, A)

        # 4. Construct the target for the loss function
        #    This will be current_q_values_all_actions, except for actions taken,
        #    where it will be td_targets_for_actions_taken.
        targets_for_loss = np.copy(current_q_values_all_actions)
        batch_indices = np.arange(BATCH_SIZE)
        targets_for_loss[batch_indices, actions] = td_targets_for_actions_taken

        # 5. Calculate Loss (MSE) and its gradient w.r.t. q_network output
        #    Loss = 0.5 * sum((Q_online - Target)^2) / BATCH_SIZE
        #    dL/dQ_online = (Q_online - Target) / BATCH_SIZE
        error = current_q_values_all_actions - targets_for_loss # (B, A)
        # loss_val = 0.5 * np.mean(np.sum(error**2, axis=1)) # For monitoring if needed
        
        grad_loss_q_values = error / BATCH_SIZE # This is dL/dQ_online

        # 6. Perform backward pass on q_network
        self.q_network.backward(grad_loss_q_values)

        # 7. Update q_network weights using the optimizer
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > EPSILON_END:
            self.epsilon -= self.epsilon_decay_value
            self.epsilon = max(EPSILON_END, self.epsilon)

        # Periodically update the target network
        if self.total_steps % TARGET_UPDATE_FREQ == 0:
            self.update_target_network()
            
    def load(self, name_prefix):
        try:
            loaded_weights = []
            for i in range(len(self.q_network.parameterized_layers)):
                w = np.load(f"{name_prefix}_layer{i}_w.npy")
                b = np.load(f"{name_prefix}_layer{i}_b.npy")
                loaded_weights.append((w,b))
            
            self.q_network.set_weights_list(loaded_weights)
            self.update_target_network()
            print(f"Model weights loaded from {name_prefix}_layerX_w/b.npy")
        except Exception as e:
            print(f"Error loading weights: {e}. Model not loaded or files missing.")

    def save(self, name_prefix):
        weights_list = self.q_network.get_weights_list()
        for i, (w,b) in enumerate(weights_list):
            np.save(f"{name_prefix}_layer{i}_w.npy", w)
            np.save(f"{name_prefix}_layer{i}_b.npy", b)
        print(f"Model weights saved to {name_prefix}_layerX_w/b.npy")

# --- Main Program Execution ---
env = gym.make('CartPole-v1')
agent = DQNAgent(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE)
all_scores = []
moving_avg_scores = []

print("Starting Training (Custom NumPy Implementation)...")
try:
    for episode in range(1, NUM_EPISODES_TRAIN + 1):
        state_tuple, info = env.reset()
        state = np.array(state_tuple, dtype=np.float32)
        total_reward = 0

        for step in range(MAX_STEPS_PER_EPISODE):
            action = agent.act(state, training=True)
            next_state_tuple, reward, terminated, truncated, info = env.step(action)
            next_state = np.array(next_state_tuple, dtype=np.float32)
            done = terminated or truncated
            
            agent.remember(state, action, reward, next_state, done) # Storing numpy arrays
            
            state = next_state
            total_reward += reward

            agent.replay()

            if done:
                break
        
        all_scores.append(total_reward)
        moving_avg = np.mean(all_scores[-100:]) if len(all_scores) >=100 else np.mean(all_scores)
        moving_avg_scores.append(moving_avg)

        if episode % 10 == 0 or episode == NUM_EPISODES_TRAIN : # Print less frequently
            print(f"Episode: {episode}/{NUM_EPISODES_TRAIN}, Score: {total_reward}, "
                  f"Epsilon: {agent.epsilon:.4f}, Moving Avg (100): {moving_avg:.2f}")

        # Adjusted solved condition for custom implementation (might be harder to reach high scores)
        if moving_avg >= MAX_STEPS_PER_EPISODE * 0.85 and len(all_scores) >= 100:
             print(f"CartPole potentially solved in {episode} episodes with custom DQN!")
             # break # Optionally stop training

except KeyboardInterrupt:
    print("Training interrupted by user.")
finally:
    agent.save("cartpole_dqn_custom_numpy_model")

# Plotting training progress
plt.figure(figsize=(12, 6))
plt.plot(all_scores, label='Score per Episode')
plt.plot(moving_avg_scores, label='Moving Average (last 100 or all if <100)', color='red', linewidth=2)
plt.title('Training Progress - CartPole DQN (Custom NumPy Implementation)')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()
plt.grid(True)
plt.savefig("training_progress_custom_numpy.png")
plt.show()

print("\nStarting Testing & GIF Generation (Custom NumPy Implementation)...")
frames = []
env_render = gym.make('CartPole-v1', render_mode='rgb_array')

agent.load("cartpole_dqn_custom_numpy_model") # Attempt to load
agent.epsilon = 0.0 # Turn off exploration for testing

for episode_test in range(NUM_EPISODES_TEST):
    state_tuple, info = env_render.reset()
    state = np.array(state_tuple, dtype=np.float32)
    total_reward_test = 0
    
    print(f"Running test episode {episode_test + 1}/{NUM_EPISODES_TEST}")
    for step_test in range(MAX_STEPS_PER_EPISODE):
        frame = env_render.render()
        frames.append(frame)
        
        action = agent.act(state, training=False)
        next_state_tuple, reward, terminated, truncated, info = env_render.step(action)
        next_state = np.array(next_state_tuple, dtype=np.float32)
        done = terminated or truncated
        
        state = next_state
        total_reward_test += reward
        
        if done:
            print(f"Test Episode {episode_test + 1} finished after {step_test + 1} steps. Score: {total_reward_test}")
            for _ in range(10): frames.append(env_render.render())
            break
    if not done: # If episode reached max steps
         print(f"Test Episode {episode_test + 1} reached max steps ({MAX_STEPS_PER_EPISODE}). Score: {total_reward_test}")

    gif_path = f"cartpole_solution_custom_numpy{episode_test}.gif"
    print(f"\nSaving GIF to {gif_path}...")
    if frames:
        imageio.mimsave(gif_path, frames, fps=30)
        print("GIF saved successfully!")
    else:
        print("No frames were captured for the GIF.")

env.close()
env_render.close()

print("\nProgram Finished.")