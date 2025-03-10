import numpy as np
import matplotlib.pyplot as plt

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))
    
    def train(self, patterns):
        """ Trénování Hopfieldovy sítě na binárních vzorech. """
        for p in patterns:
            self.weights += np.outer(p, p)
        np.fill_diagonal(self.weights, 0)  # Nastavení diagonály na 0

    def recall(self, pattern, steps=5, mode='synchronous'):
        """ Obnova vzoru synchronně nebo asynchronně. """
        recalled_pattern = np.copy(pattern)
        for _ in range(steps):
            if mode == 'synchronous':
                recalled_pattern = np.sign(self.weights @ recalled_pattern)
            elif mode == 'asynchronous':
                for i in range(self.size):
                    recalled_pattern[i] = np.sign(np.dot(self.weights[i], recalled_pattern))
        return recalled_pattern

# Definice vzorů (3x3 binární matice jako vektor)
patterns = np.array([
    [-1, -1, 1, -1, 1, 1, -1, 1, 1],  # První vzor
    [1, -1, -1, 1, -1, 1, 1, -1, -1]  # Druhý vzor
])

# Inicializace Hopfieldovy sítě
hopfield_net = HopfieldNetwork(size=9)
hopfield_net.train(patterns)

# Poškozený vzor (náhodné změny)
noisy_pattern = np.array([-1, -1, 1, -1, -1, 1, -1, 1, -1])

# Obnova vzoru
recalled_pattern_sync = hopfield_net.recall(noisy_pattern, mode='synchronous')
recalled_pattern_async = hopfield_net.recall(noisy_pattern, mode='asynchronous')

# Funkce pro vizualizaci vzorů
def plot_pattern(pattern, title):
    plt.imshow(pattern.reshape((3, 3)), cmap="gray")
    plt.title(title)
    plt.axis("off")

# Vykreslení vzorů
plt.figure(figsize=(8, 4))
plt.subplot(1, 3, 1)
plot_pattern(noisy_pattern, "Noisy Input")
plt.subplot(1, 3, 2)
plot_pattern(recalled_pattern_sync, "Recalled (Sync)")
plt.subplot(1, 3, 3)
plot_pattern(recalled_pattern_async, "Recalled (Async)")
plt.show()
