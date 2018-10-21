from matplotlib import pyplot as plt
from matplotlib import animation as anim
import matplotlib.cm as cm
from source.clusterizer import Clusterizer
import numpy as np


n, c = 5, 25
clusterizer = Clusterizer(c)

colors = cm.rainbow(np.linspace(0, 1, c))
fig, ax = plt.subplots(1)
ax.set_title(f"2D clusterizer, {c} clusters")
sc = ax.scatter([], [], color=colors)
plt.xlim(0,1)
plt.ylim(0,1)

def animate(i):

    for _ in range(50):
        d = np.random.random(size=2)
        clusterizer.fit(d, learn_rate=0.1 , random_rate=0)

    sc.set_offsets(clusterizer.centers)


ani = anim.FuncAnimation(fig, animate, frames=300, repeat_delay=5000, repeat=True)

ani.save('../pictures/picture_01.gif', writer='imagemagick', fps=10)

