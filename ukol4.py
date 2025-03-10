import numpy as np
import random
import matplotlib.pyplot as plt

# Parametry prostředí
GRID_SIZE = 5
ACTIONS = ['up', 'down', 'left', 'right']
ACTION_IDX = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
NUM_EPISODES = 500
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPSILON = 0.1  # Prozkoumávání vs. využívání

# Inicializace Q-matice
Q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))

# Umístění sýra (cíl)
goal_position = (4, 4)

# Funkce pro výběr akce (epsilon-greedy)
def choose_action(state):
    if random.uniform(0, 1) < EPSILON:
        return random.choice(ACTIONS)  # Průzkum
    else:
        return ACTIONS[np.argmax(Q_table[state[0], state[1]])]  # Využití znalostí

# Funkce pro aktualizaci stavu po provedení akce
def get_next_state(state, action):
    x, y = state
    if action == 'up':
        x = max(0, x - 1)
    elif action == 'down':
        x = min(GRID_SIZE - 1, x + 1)
    elif action == 'left':
        y = max(0, y - 1)
    elif action == 'right':
        y = min(GRID_SIZE - 1, y + 1)
    return (x, y)

# Trénování agenta pomocí Q-learningu
rewards = []
for episode in range(NUM_EPISODES):
    state = (0, 0)  # Startovní pozice agenta
    total_reward = 0

    while state != goal_position:
        action = choose_action(state)
        next_state = get_next_state(state, action)

        # Odměna za dosažení sýra
        reward = 1 if next_state == goal_position else -0.01
        total_reward += reward

        # Q-learning update rule
        x, y = state
        nx, ny = next_state
        a_idx = ACTION_IDX[action]
        best_next_q = np.max(Q_table[nx, ny])
        Q_table[x, y, a_idx] += LEARNING_RATE * (reward + DISCOUNT_FACTOR * best_next_q - Q_table[x, y, a_idx])

        state = next_state  # Přesun na nový stav

    rewards.append(total_reward)  # Uložení odměny za epizodu

# Vizualizace průběhu učení
plt.plot(rewards)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Q-learning Training Progress')
plt.show()

# Zobrazení naučené Q-matice
print("Final Q-table:")
print(Q_table)
