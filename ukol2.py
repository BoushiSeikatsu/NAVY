import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Definice neuronové sítě (MLP)
model = keras.Sequential([
    keras.layers.Dense(4, input_dim=2, activation='relu'),  # Skrytá vrstva
    keras.layers.Dense(1, activation='sigmoid')  # Výstupní vrstva
])

# Kompilace modelu
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Trénování modelu
history = model.fit(X, y, epochs=500, verbose=0)

# Vyhodnocení modelu
predictions = (model.predict(X) > 0.5).astype(int)
print("Predictions:", predictions.flatten())

# Vizualizace rozhodovací hranice
def plot_decision_boundary(model, X, y):
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100), np.linspace(-0.5, 1.5, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = model.predict(grid)
    preds = preds.reshape(xx.shape)

    plt.contourf(xx, yy, preds, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap="bwr", edgecolors='k')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Decision Boundary for XOR Problem")
    plt.show()

plot_decision_boundary(model, X, y)

# Vizualizace průběhu trénování
plt.plot(history.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
