import numpy as np

# Jelikož výsledky budou od 0 do 1 tak můžu použít sigmoid funkci
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# XOR Tabulka
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Inicializace náhodných vah a vytvoření struktury neuronové sítě 
np.random.seed(1)
input_layer_neurons = 2
hidden_layer_neurons = 2  # Potřebujeme alespoň 2 neurony 
output_layer_neurons = 1
weights_input_hidden = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
weights_hidden_output = np.random.uniform(size=(hidden_layer_neurons, output_layer_neurons))
bias_hidden = np.random.uniform(size=(1, hidden_layer_neurons))
bias_output = np.random.uniform(size=(1, output_layer_neurons))

print("-----------------------------------------------------")
print("Weight and bias before the learning phase:")
print(f"neuron_hidden1.weights {weights_input_hidden[0]}")
print(f"neuron_hidden2.weights {weights_input_hidden[1]}")
print(f"neuron_output.weights {weights_hidden_output.flatten()}")
print(f"neuron_hidden1.bias {bias_hidden[0,0]}")
print(f"neuron_hidden2.bias {bias_hidden[0,1]}")
print(f"neuron_output.bias {bias_output[0,0]}")
print("-----------------------------------------------------")
print("\nLearning in progress..\n")

# Nastavení trénovacích parametrů 
learning_rate = 0.5
epochs = 10000

# Trénování
# Vždycky se nejdříve spočítá aktivační funkce a přičte se bias
# Potom převedeme output přes naši sigmoid funkci
# Potom jdeme do output layer, která má za úkol převést hodnoty ANN do stejného měřítka jako jsou reálné predikované hodnoty
# Pak se spočítá loss funkce a skrz back propagation se upraví váhy a bias
# Pak následuje další epoch
for epoch in range(epochs):
    # Forward propagation
    hidden_layer_activation = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_activation)
    
    output_layer_activation = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    output = sigmoid(output_layer_activation)
    
    error = y - output
    
    # Backpropagation
    d_output = error * sigmoid_derivative(output)
    
    error_hidden_layer = d_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # Update vah a biasu 
    weights_hidden_output += hidden_layer_output.T.dot(d_output) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate
    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

print("\n-----------------------------------------------------")
print("Weight and bias after the learning phase:")
print(f"neuron_hidden1.weights {weights_input_hidden[0]}")
print(f"neuron_hidden2.weights {weights_input_hidden[1]}")
print(f"neuron_output.weights {weights_hidden_output.flatten()}")
print(f"neuron_hidden1.bias {bias_hidden[0,0]}")
print(f"neuron_hidden2.bias {bias_hidden[0,1]}")
print(f"neuron_output.bias {bias_output[0,0]}")
print("-----------------------------------------------------")

# Testování
print("\nTesting in progress..\n")
correct = 0
total = len(X)
for i in range(total):
    hidden_layer_activation = np.dot(X[i], weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_activation)
    
    output_layer_activation = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    prediction = sigmoid(output_layer_activation)
    
    expected = y[i][0]
    is_correct = (round(prediction[0,0]) == expected)
    correct += is_correct
    
    print(f"Guess {prediction[0,0]:.12f}   Expected output {expected}   Is it equal? {is_correct}")

accuracy = (correct / total) * 100
print(f"\nSuccess is {accuracy:.1f} %")
print("-----------------------------------------------------")
