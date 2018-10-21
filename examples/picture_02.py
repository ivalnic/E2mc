from matplotlib import pyplot as plt
from matplotlib import animation as anim
import matplotlib.cm as cm
from source.clusterizer import Clusterizer
import numpy as np

n, c = 5, 15
clusterizer = Clusterizer(c)
x_train = np.load("../mnist/Mnist_X_train.npy")

d1, d2, d3 = x_train.shape

fig, ax = plt.subplots(1, figsize=(c/2, 1))
ax.set_title(f"{d2*d3}D clusterizer, {c} clusters")
im = ax.imshow(np.zeros((d2, c*d3)), vmax=255)

ix = 0

def animate(i):
    global ix
    for _ in range(50):
        clusterizer.fit(x_train[ix].flatten(), learn_rate=0.05)
        ix+=1
    im.set_data(
        clusterizer.centers.reshape(-1, d2, d3).swapaxes(0,1).reshape(d2, -1))


ani = anim.FuncAnimation(fig, animate, frames=100, repeat_delay=5000, repeat=True)

ani.save('../pictures/picture_02.gif', writer='imagemagick', fps=10)

