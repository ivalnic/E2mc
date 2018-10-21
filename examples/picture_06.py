from matplotlib import pyplot as plt
from matplotlib import animation as anim
from source.meta_clusterizer import MetaClusterizer
import numpy as np

n, c = 2, 10
x_train = np.load("../mnist/Mnist_X_train.npy") / 255
d1, d2, d3 = x_train.shape

x_train = x_train.reshape(60000//2, 1, 2, d2, d3).swapaxes(2,3).reshape(60000//2, d2, d3*2)
d1, d2, d3 = x_train.shape

meta = MetaClusterizer(shape=(d2*d3,), n_meta=2, n_clusters=c)


fig = plt.figure(figsize=(c/2 + 2, 3))

ax1 = plt.subplot2grid((2, c+3), (0,0), colspan=c)
ax2 = plt.subplot2grid((2, c+3), (1,0), colspan=c)
axm1 = plt.subplot2grid((2, c+3), (0, c+1), colspan=2)
axm2 = plt.subplot2grid((2, c+3), (1, c+1), colspan=2)


ax1.set_title(f"{d2*d3}D clusterizer, {c} clusters")
ax2.set_title(f"{d2*d3}D clusterizer, {c} clusters")
axm1.set_title(f"Mask 1")
axm2.set_title(f"Mask 2")


im1 = ax1.imshow(np.zeros((d2, c*d3)), vmax=1)
im2 = ax2.imshow(np.zeros((d2, c*d3)), vmax=1)

imm1 = axm1.imshow(np.zeros((d2, d3)), vmax=1)
imm2 = axm2.imshow(np.zeros((d2, d3)), vmax=1)

ix = 0
b = 100

def animate(i):
    global ix

    for _ in range (100):
        meta.fit(x_train[ix:ix+b].reshape(b, -1), learn_rate= 0.2, clearn_rate=0.02, random_rate=0.01)
        ix= (ix+b)%d1

    im1.set_data(
        meta.clusters[0].centers.reshape(-1, d2, d3).swapaxes(0,1).reshape(d2, -1))
    im2.set_data(
        meta.clusters[1].centers.reshape(-1, d2, d3).swapaxes(0,1).reshape(d2, -1))

    imm1.set_data(meta.masks[0].reshape(d2, d3))
    imm2.set_data(meta.masks[1].reshape(d2, d3))

    plt.pause(0.1)


ani = anim.FuncAnimation(fig, animate, frames=20, repeat_delay=5000, repeat=True)

ani.save('../pictures/picture_06.gif', writer='imagemagick', fps=3)

