import gym
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

# Vytvoření prostředí CartPole
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
num_actions = env.action_space.n

# Definice neuronové sítě pro Q-learning
def build_model(state_size, num_actions):
    model = Sequential([
        Flatten(input_shape=(1, state_size)),
        Dense(24, activation='relu'),
        Dense(24, activation='relu'),
        Dense(num_actions, activation='linear')
    ])
    return model

# Inicializace modelu
model = build_model(state_size, num_actions)
# Definice agenta DQN
def build_agent(model, num_actions):
    policy = EpsGreedyQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   nb_actions=num_actions, nb_steps_warmup=10,
                   target_model_update=1e-2)
    return dqn

# Inicializace DQN agenta
dqn = build_agent(model, num_actions)
dqn.compile(Adam(learning_rate=0.001), metrics=['mae'])

# Trénování modelu
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

# Vyhodnocení naučeného agenta
dqn.test(env, nb_episodes=5, visualize=True)
