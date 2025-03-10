import matplotlib.pyplot as plt
import numpy as np

def l_system(axiom, rules, iterations):
    """Generuje řetězec L-systému po daném počtu iterací."""
    result = axiom
    for _ in range(iterations):
        result = ''.join(rules.get(char, char) for char in result)
    return result

def draw_l_system(instructions, angle, length):
    """Vykreslí L-systém na základě instrukcí."""
    stack = []
    position = np.array([0, 0])
    direction = np.array([0, 1])
    points = [position.copy()]

    for command in instructions:
        if command == 'F':
            position = position.astype(float)  # Convert to float first
            position += direction * length
            points.append(position.copy())
        elif command == '+':
            direction = np.dot(rotation_matrix(angle), direction)
        elif command == '-':
            direction = np.dot(rotation_matrix(-angle), direction)
        elif command == '[':
            stack.append((position.copy(), direction.copy()))
        elif command == ']':
            position, direction = stack.pop()
            points.append(None)  # Označuje přerušení čáry

    return points

def rotation_matrix(theta):
    """Vytvoří 2D rotační matici pro úhel theta."""
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])

def plot_l_system(points, title):
    """Vykreslí body L-systému."""
    plt.figure(figsize=(8, 8))
    current_segment = []
    for point in points:
        if point is not None:
            current_segment.append(point)
        elif current_segment:
            current_segment = np.array(current_segment)
            plt.plot(current_segment[:, 0], current_segment[:, 1], 'k-')
            current_segment = []
    if current_segment:
        current_segment = np.array(current_segment)
        plt.plot(current_segment[:, 0], current_segment[:, 1], 'k-')
    plt.title(title)
    plt.axis('equal')
    plt.axis('off')
    plt.show()

# Definice L-systémů
l_systems = [
    {
        'axiom': 'F+F+F+F',
        'rules': {'F': 'F+F-F-FF+F+F-F'},
        'angle': np.pi / 2,
        'iterations': 2,
        'title': 'L-systém 1'
    },
    {
        'axiom': 'F++F++F',
        'rules': {'F': 'F+F--F+F'},
        'angle': np.pi / 3,
        'iterations': 3,
        'title': 'L-systém 2'
    },
    {
        'axiom': 'F',
        'rules': {'F': 'F[+F]F[-F]F'},
        'angle': np.pi / 7,
        'iterations': 5,
        'title': 'L-systém 3'
    },
    {
        'axiom': 'F',
        'rules': {'F': 'FF+[+F-F-F]-[-F+F+F]'},
        'angle': np.pi / 8,
        'iterations': 4,
        'title': 'L-systém 4'
    }
]

# Vykreslení L-systémů
for system in l_systems:
    instructions = l_system(system['axiom'], system['rules'], system['iterations'])
    points = draw_l_system(instructions, system['angle'], length=5)
    plot_l_system(points, system['title'])
