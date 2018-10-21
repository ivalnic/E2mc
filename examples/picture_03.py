from matplotlib import pyplot as plt
from matplotlib import animation as anim
import matplotlib.cm as cm
from source.clusterizer import Clusterizer
import numpy as np

n, c = 5, 10
clusterizer1 = Clusterizer(c)
clusterizer2 = Clusterizer(c)
x_train = np.load("../mnist/Mnist_X_train.npy")

d1, d2, d3 = x_train.shape

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(c/2, 2))
ax1.set_title(f"{d2*d3}D clusterizer, {c} clusters")
ax2.set_title(f"{d2*d3}D clusterizer, {c} clusters")
im1 = ax1.imshow(np.zeros((d2, c*d3)), vmax=255)
im2 = ax2.imshow(np.zeros((d2, c*d3)), vmax=255)

ix = 0

def animate(i):
    global ix
    for _ in range(50):
        clusterizer1.fit(x_train[ix].flatten(), learn_rate=0.05, random_rate=0.01)
        clusterizer2.fit(x_train[ix].flatten(), learn_rate=0.05, random_rate=0.01)
        ix+=1
    im1.set_data(
        clusterizer1.centers.reshape(-1, d2, d3).swapaxes(0,1).reshape(d2, -1))
    im2.set_data(
        clusterizer2.centers.reshape(-1, d2, d3).swapaxes(0,1).reshape(d2, -1))


ani = anim.FuncAnimation(fig, animate, frames=100, repeat_delay=5000, repeat=True)

ani.save('../pictures/picture_03.gif', writer='imagemagick', fps=10)

