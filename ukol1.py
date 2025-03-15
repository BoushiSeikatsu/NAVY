import numpy as np
import matplotlib.pyplot as plt

# Nejdříve vygeneruji 100 náhodných bodů
np.random.seed(13)
X = np.random.uniform(-20, 20, (100, 2))

# Určí zda bod leží nad, pod nebo na přímce
def classify_point(x, y):
    line_y = 3 * x + 2
    if y > line_y:
        return 1  # Nad přímkou
    elif y < line_y:
        return -1  # Pod přímkou
    else:
        return 0  # Na přímce

# Určím labels 
y_labels = np.array([classify_point(x[0], x[1]) for x in X])

# Inicializace váhového vektoru a biasu
weights = np.random.rand(2)
bias = np.random.rand(1)
learning_rate = 0.01
epochs = 100

# Trénování perceptronu
# Pro každý vygenerovaný bod zvlášť vypočítáme aktivační funkci, potom porovnáme jestli se predikovaná hodnota liší od labelu a podle toho upravíme váhy a bias
for epoch in range(epochs):
    for i, x in enumerate(X):
        y_pred = np.dot(x, weights) + bias
        y_pred = np.sign(y_pred)  # Aplikuje signovou aktivační funkci
        
        if y_pred != y_labels[i]:  # Aktualizace vah a biasu při chybě
            weights += learning_rate * y_labels[i] * x
            bias += learning_rate * y_labels[i]

# Vizualizace výsledků
plt.figure(figsize=(8, 6))

# Přímka y = 3x + 2
x_line = np.linspace(-20, 20, 100)
y_line = 3 * x_line + 2
plt.plot(x_line, y_line, 'k-', label="y = 3x + 2")

# Rozdělení bodů podle predikovaných tříd
for i, x in enumerate(X):
    if y_labels[i] == 1:
        plt.scatter(x[0], x[1], color='blue', marker='o', label="Nad carou" if i == 0 else "")
    elif y_labels[i] == -1:
        plt.scatter(x[0], x[1], color='red', marker='o', label="Pod carou" if i == 0 else "")
    else:
        plt.scatter(x[0], x[1], color='green', marker='x', label="Na care" if i == 0 else "")

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Perceptron - Predikce umístění vůči y = 3x + 2")
plt.grid()
plt.show()
