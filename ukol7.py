import numpy as np
import matplotlib.pyplot as plt

# Definice transformací pro první model
transformations_model_1 = [
    {'a': 0.00, 'b': 0.00, 'c': 0.01, 'd': 0.00, 'e': 0.26, 'f': 0.00, 'g': 0.00, 'h': 0.00, 'i': 0.05, 'j': 0.00, 'k': 0.00, 'l': 0.00},
    {'a': 0.20, 'b': -0.26, 'c': -0.01, 'd': 0.23, 'e': 0.22, 'f': -0.07, 'g': 0.07, 'h': 0.00, 'i': 0.24, 'j': 0.00, 'k': 0.80, 'l': 0.00},
    {'a': -0.25, 'b': 0.28, 'c': 0.01, 'd': 0.26, 'e': 0.24, 'f': -0.07, 'g': 0.07, 'h': 0.00, 'i': 0.24, 'j': 0.00, 'k': 0.22, 'l': 0.00},
    {'a': 0.85, 'b': 0.04, 'c': -0.01, 'd': -0.04, 'e': 0.85, 'f': 0.09, 'g': 0.00, 'h': 0.08, 'i': 0.84, 'j': 0.00, 'k': 0.80, 'l': 0.00}
]

# Definice transformací pro druhý model
transformations_model_2 = [
    {'a': 0.05, 'b': 0.00, 'c': 0.00, 'd': 0.00, 'e': 0.60, 'f': 0.00, 'g': 0.00, 'h': 0.00, 'i': 0.05, 'j': 0.00, 'k': 0.00, 'l': 0.00},
    {'a': 0.45, 'b': -0.22, 'c': 0.22, 'd': 0.22, 'e': 0.45, 'f': 0.22, 'g': -0.22, 'h': 0.22, 'i': -0.45, 'j': 0.00, 'k': 1.00, 'l': 0.00},
    {'a': -0.45, 'b': 0.22, 'c': -0.22, 'd': 0.22, 'e': 0.45, 'f': 0.22, 'g': 0.22, 'h': -0.22, 'i': 0.45, 'j': 0.00, 'k': 1.25, 'l': 0.00},
    {'a': 0.49, 'b': -0.08, 'c': 0.08, 'd': 0.08, 'e': 0.49, 'f': 0.08, 'g': 0.08, 'h': -0.08, 'i': 0.49, 'j': 0.00, 'k': 2.00, 'l': 0.00}
]

def apply_transformation(x, y, z, transformation):
    x_new = (transformation['a'] * x + transformation['b'] * y + transformation['c'] * z + transformation['j'])
    y_new = (transformation['d'] * x + transformation['e'] * y + transformation['f'] * z + transformation['k'])
    z_new = (transformation['g'] * x + transformation['h'] * y + transformation['i'] * z + transformation['l'])
    return x_new, y_new, z_new

def generate_ifs_fractal(transformations, iterations=100000):
    x, y, z = 0, 0, 0  # Počáteční bod
    points = []

    for _ in range(iterations):
        transformation = np.random.choice(transformations)
        x, y, z = apply_transformation(x, y, z, transformation)
        points.append((x, y))

    return points

def plot_fractal(points, title):
    x_vals, y_vals = zip(*points)
#    plt.figure(figsize
#::contentReference[oaicite:9]{index=9} GPT STOPPED HERE
 
