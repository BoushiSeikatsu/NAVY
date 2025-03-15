import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

def mandelbrot(c, max_iter):
    z = 0
    n = 0
    while abs(z) <= 2 and n < max_iter:
        z = z*z + c
        n += 1
    return n

def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    n3 = np.empty((width, height))
    for i in range(width):
        for j in range(height):
            n3[i, j] = mandelbrot(r1[i] + 1j*r2[j], max_iter)
    return (r1, r2, n3)

def display(xmin, xmax, ymin, ymax, width=10, height=10, max_iter=256):
    dpi = 80
    img_width = dpi * width
    img_height = dpi * height
    x, y, z = mandelbrot_set(xmin, xmax, ymin, ymax, img_width, img_height, max_iter)
    
    # Normalize z
    z = z.T
    norm = plt.Normalize(vmin=z.min(), vmax=z.max())
    z = norm(z)
    
    # Apply HSV colormap
    hsv = np.zeros((z.shape[0], z.shape[1], 3))
    hsv[..., 0] = z
    hsv[..., 1] = 1
    hsv[..., 2] = z < 1
    rgb = hsv_to_rgb(hsv)
    
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    ax.imshow(rgb, extent=(xmin, xmax, ymin, ymax))
    ax.set_title("Mandelbrotova množina")
    plt.show()

# Původní zobrazení
display(-2.0, 1.0, -1.0, 1.0)

# Zoom na konkrétní oblast
display(-0.75, -0.74, 0.1, 0.11)
