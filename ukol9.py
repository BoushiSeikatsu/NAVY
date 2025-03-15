import matplotlib.pyplot as plt
import numpy as np

def midpoint_displacement(start, end, roughness, detail, vertical_displacement=None):
    if vertical_displacement is None:
        vertical_displacement = (start[1] + end[1]) / 2

    points = [start, end]
    iteration = 1

    while iteration <= detail:
        new_points = []
        for i in range(len(points) - 1):
            midpoint = [(points[i][0] + points[i + 1][0]) / 2,
                        (points[i][1] + points[i + 1][1]) / 2]
            midpoint[1] += np.random.uniform(-vertical_displacement, vertical_displacement)
            new_points.extend([points[i], midpoint])
        new_points.append(points[-1])
        points = new_points
        vertical_displacement *= 2 ** (-roughness)
        iteration += 1

    return points

def plot_terrain(points, color='brown'):
    x, y = zip(*points)
    plt.plot(x, y, color=color)
    plt.fill_between(x, y, min(y) - 10, color=color, alpha=0.5)

# Parametry terénu
start_point = [0, 0]
end_point = [100, 0]
roughness = 0.5
detail = 10

# Generování různých terénů
terrain1 = midpoint_displacement(start_point, end_point, roughness, detail, vertical_displacement=20)
terrain2 = midpoint_displacement(start_point, end_point, roughness, detail, vertical_displacement=10)
terrain3 = midpoint_displacement(start_point, end_point, roughness, detail, vertical_displacement=5)

# Vykreslení terénů
plt.figure(figsize=(10, 6))
plot_terrain(terrain1, color='saddlebrown')
plot_terrain(terrain2, color='forestgreen')
plot_terrain(terrain3, color='darkgreen')
plt.title('Generování 2D krajiny pomocí fraktální geometrie')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
