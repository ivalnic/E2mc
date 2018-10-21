from matplotlib import pyplot as plt
from matplotlib import animation as anim
from source.meta_clusterizer import MetaClusterizer
import numpy as np


x_train = np.load("../mnist/Mnist_X_train.npy") / 255
x_train = x_train.reshape(60000//2, 1, 2, 28, 28).swapaxes(2,3).reshape(60000//2, 28, 56)


plt.imshow(x_train[0])
plt.savefig('../pictures/picture_05.jpeg')
