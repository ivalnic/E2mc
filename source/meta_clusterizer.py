import numpy as np
from source.clusterizer import Clusterizer
import pickle

class MetaClusterizer:
    def __init__(self, n_meta  = 2, n_clusters  = 20, shape = (10,)):
        self.n_meta  = n_meta
        self.n_clusters  = n_clusters
        self.masks = np.ones(((n_meta,) + shape)) / n_meta
        self.shape = shape

        self.clusters = [Clusterizer(n_clusters=self.n_clusters, ) for _ in self.masks]
        self.learn_state = 1

    def fit(self, x, learn_rate = 0.5, random_rate = 0.5, clearn_rate=0.3):

        self.batch_error = np.zeros(self.masks.shape)

        errors = []
        for pattern, cluster in  zip(self.masks, self.clusters):

            xp = x.flatten() * pattern

            labels = cluster.fit(xp, learn_rate=clearn_rate, random_rate= random_rate)
            bests = cluster.centers[labels]

            error = np.abs((xp - bests))
            errors.append(error)

        self.batch_error += np.asarray(errors)


        self.masks -= self.batch_error * learn_rate
        self.masks[self.masks < 0.001] =  0.001

        s = self.masks.reshape((self.n_meta,) + self.shape).sum(axis=0)
        self.masks /= s

        self.learn_state = self.masks.max(axis=0).mean()

