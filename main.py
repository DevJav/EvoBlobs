import copy
from blob import Blob
import random
from matplotlib import pyplot as plt
from constants import *
from scipy.spatial import KDTree
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def main():
    # Initialize the simulation environment
    foods = [(random.uniform(MIN_X, MAX_X), random.uniform(MIN_Y, MAX_Y)) for _ in range(N_FOOD)]
    blobs = [Blob() for _ in range(N_BLOBS)]

    # Setup the plot
    plt.ion()
    fig, ax = plt.subplots(num="Simulation")
    fig.patch.set_facecolor('black')  # Set the background color of the figure
    ax.set_facecolor('black')  # Set the background color of the plot
    ax.set_xlim(MIN_X, MAX_X)
    ax.set_ylim(MIN_Y, MAX_Y)
    blob_scatter = ax.scatter([], [], s=2)
    food_scatter = ax.scatter([], [], color='orange', s=1, marker='v')
    texts = []

    id_to_color = {}
    color_map = cm.get_cmap('hsv')  # o 'tab20', 'nipy_spectral', etc.

    for i in range(10000000):
        # Build KDTree for current food positions
        if foods:
            food_tree = KDTree(foods)

        new_blobs = []
        for blob in blobs:
            blob.step(foods)

            # If blob touches the edge of the screen, wrap around
            if blob.x < MIN_X:
                blob.x = MAX_X
            elif blob.x > MAX_X:
                blob.x = MIN_X
            if blob.y < MIN_Y:
                blob.y = MAX_Y
            elif blob.y > MAX_Y:
                blob.y = MIN_Y

            # Check for food collision (using KDTree)
            if foods:
                nearby = food_tree.query_ball_point([blob.x, blob.y], r=10)
                if nearby:
                    food_index = nearby[0]
                    food_pos = food_tree.data[food_index]
                    
                    try:
                        real_index = foods.index(tuple(food_pos))
                        eaten = foods.pop(real_index)
                        new_blob = blob.eat()
                        if new_blob:
                            new_blobs.append(new_blob)
                    except ValueError:
                        pass  # La comida ya fue comida, ignorar

        # Add new blobs to the population
        blobs.extend(new_blobs)
        # Update plot
        blob_positions = [(blob.x, blob.y) for blob in blobs]
        # blob_colors = ['green' if blob.life > 0 else 'red' for blob in blobs]
        # Asignar un color único a cada id si no lo tenía ya
        for blob in blobs:
            if blob.id[:7] not in id_to_color:
                hash_val = abs(hash(blob.id[:7])) % 1000  # Un número entre 0 y 999
                id_to_color[blob.id[:7]] = color_map(hash_val / 1000)  # Normalizado [0,1]

        blob_colors = [id_to_color[blob.id[:7]] for blob in blobs]

        blob_sizes = [max(1, blob.food_eaten*4) for blob in blobs]  # tamaño mínimo 20, escala por comida

        blob_scatter.set_offsets(blob_positions)
        blob_scatter.set_color(blob_colors)
        blob_scatter.set_sizes(blob_sizes)

        food_scatter.set_offsets(foods)

        # Remove previous texts
        for t in texts:
            t.remove()
        texts.clear()

        # Write top 5 blobs
        top_blobs = sorted(blobs, key=lambda b: b.food_eaten, reverse=True)[:5]
        for j, blob in enumerate(top_blobs):
            text = ax.text(
                0.5, 0.95 - j * 0.07,
                f"Blob {j+1}: {blob.id} | Food eaten: {blob.food_eaten}",
                color='white',
                fontsize=8,
                ha='center',
                transform=ax.transAxes,
                bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.3')
            )
            texts.append(text)

        fig.canvas.draw_idle()
        plt.pause(0.01)

        if i % 100 == 0:
            print(f"Step {i}")

        # if i % 500 == 0 and i > 0:
        #     blobs = evolve(blobs)

        # Remove dead blobs
        blobs = [blob for blob in blobs if blob.life > 0]
        if not blobs:
            print("All blobs have died.")
            break

        # Reespawn food at given rate
        spawn_food = 0.3  # 10% chance to spawn food each step
        if random.random() < spawn_food or len(foods) < N_FOOD // 4:
            for _ in range(5):
                if len(foods) < N_FOOD:
                    food_pos = (random.uniform(MIN_X, MAX_X), random.uniform(MIN_Y, MAX_Y))
                    foods.append(food_pos)

def evolve(blobs):
    # 1. Ordenar por rendimiento
    top_blobs = sorted(blobs, key=lambda b: b.food_eaten, reverse=True)[:10]

    # 2. Crear nuevos blobs a partir de los mejores
    new_generation = []
    for blob in top_blobs:
        for _ in range(15):  # Cada uno tiene 5 hijos
            child = blob.reproduce()
            new_generation.append(child)

    # 3. (Opcional) mantener a los padres
    # new_generation += top_blobs

    # 4. Reemplazar población
    blobs = new_generation

    return blobs

if __name__ == "__main__":
    main()
