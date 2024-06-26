# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 06:03:14 2024

@author: JannisKammeier
"""
import matplotlib.pyplot as plt
from ActiveLearner import ActiveLearner
from time import time

al:ActiveLearner = ActiveLearner(numInitDatapoints=500)

# choose one of the three sampling strategies
labelingOrder = al.calculateMargin()
# labelingOrder = al.calculateEntropies()
# labelingOrder = al.calculateLeastConfident()

sortImg = al.xTrainUnlabeled[labelingOrder]


# plot the three most informative images
plt.figure(figsize=(28*3, 28))
for i in range (3):
    plt.subplot(1, 3, i+1)
    plt.imshow(255-sortImg[i], cmap="gray")
    plt.axis("off")

plt.suptitle("most informative images", fontsize=150)


plt.savefig(f"mostInfo_{time()}.png")
plt.show()


# plot the three least informative images
plt.figure(figsize=(28*3, 28))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(255-sortImg[-(i+1)], cmap="gray")
    plt.axis("off")

plt.suptitle("least informative images", fontsize=150)


plt.savefig(f"leastInfo_{time()}.png")
plt.show()


del al
    
