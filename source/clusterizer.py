import numpy as np


class Clusterizer:

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

        self.centers = None


    def fit(self, data, learn_rate = 0.01, random_rate = 0.01):

        if self.centers is None:
            self.centers = np.zeros((self.n_clusters,) + data.shape)

        diff = self.centers - data[None]

        error = np.abs(diff)
        error = error.mean(axis=1)

        best = error.argmin()

        if np.random.random() < random_rate:
            best = np.random.randint(error.size)

        self.centers[best] -= diff[best] * learn_rate

        return best

if __name__ == "__main__":

    from matplotlib import pyplot as plt

    c = Clusterizer(10)
    x_train = np.load("Mnist_X_train.npy")

    for i, x in enumerate(x_train):
        c.fit(x.flatten(), learn_rate=0.02)

        if i % 100 == 0:

            plt.cla()
            plt.imshow(c.centers.reshape(-1,28,28).swapaxes(0,1).reshape(28,-1))
            plt.title(f"{i} samples")
            plt.pause(0.1)