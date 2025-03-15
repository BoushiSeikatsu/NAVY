import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

# 1. Implementace logistické mapy
def logistic_map(a, x):
    return a * x * (1 - x)

# 2. Vizualizace bifurkačního diagramu
def bifurcation_diagram(a_values, x0, iterations, last):
    x = x0 * np.ones(len(a_values))
    for i in range(iterations):
        x = logistic_map(a_values, x)
        if i >= (iterations - last):
            plt.plot(a_values, x, ',k', alpha=0.25)
    plt.title("Bifurkační diagram")
    plt.xlabel("a")
    plt.ylabel("x")
    plt.show()

a_values = np.linspace(0, 4.0, 1000)
bifurcation_diagram(a_values, x0=0.5, iterations=1000, last=100)

# 3. Predikce pomocí neuronové sítě
# Generování tréninkových dat
train_size = 10000
a_train = np.random.uniform(0, 4, train_size)
x_train = np.random.uniform(0, 1, train_size)
y_train = logistic_map(a_train, x_train)

# Trénink neuronové sítě
nn = MLPRegressor(hidden_layer_sizes=(10,), activation='tanh', solver='adam', max_iter=2000)
nn.fit(np.vstack([a_train, x_train]).T, y_train)

# 4. Vizualizace predikovaného bifurkačního diagramu
def predicted_bifurcation_diagram(nn, a_values, x0, iterations, last):
    x = x0 * np.ones(len(a_values))
    for i in range(iterations):
        x = nn.predict(np.vstack([a_values, x]).T)
        if i >= (iterations - last):
            plt.plot(a_values, x, ',r', alpha=0.25)
    plt.title("Predikovaný bifurkační diagram")
    plt.xlabel("a")
    plt.ylabel("x")
    plt.show()

predicted_bifurcation_diagram(nn, a_values, x0=0.5, iterations=1000, last=100)
