import numpy as np


from source.meta_clusterizer import MetaClusterizer
import matplotlib.animation as animation

from matplotlib import pyplot as plt

m, n = 5, 5

learn_rate =  0.5
learn_crate =  0.3
learn_random = 0.02

plot_rate = m * 10

meta = MetaClusterizer(n_meta=m, n_clusters=m , shape = (m * n,))

plt.ion()
fig, axim = plt.subplots(1, 1, figsize=(10,3))
axim.imshow(meta.masks, vmax=2 / m)
axim.set_title(f"Meta-clusterizer. {m} meta clusters. {n} clusters in each.")

ims = []
i=0
count_down = 3

while True:

    d = np.zeros((m, n))
    d[np.arange(m, dtype=int), np.random.randint(n, size=m)] = 1

    meta.fit(d.reshape(1, -1), learn_rate=learn_rate, random_rate=learn_random, clearn_rate=learn_crate)



    if i%plot_rate == 0 and i >0:
        im1 = axim.imshow(meta.masks)

        fig.canvas.draw()
        fig.canvas.flush_events()

        ims.append((im1,))

        best = meta.masks.argmax(axis=0).reshape(-1, n).T
        if (best == best[0, :]).all():
            count_down -=1
        else:
            count_down = 3
        if count_down <=0:
            break

    i+=1


im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000,
                                   blit=True)
im_ani.save('../pictures/picture_07.gif', writer='imagemagick', fps=3)