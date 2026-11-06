import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def graph_images(images):
    amount = len(images)
    rows = amount//5
    if amount % 5 != 0:
        rows = rows + 1
    plt.figure(figsize=(11, rows*2))
    for i in range(amount):
        plt.subplot(rows, 5, i+1)
        image = images[i].reshape(28, 28)
        plt.imshow(image, cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


