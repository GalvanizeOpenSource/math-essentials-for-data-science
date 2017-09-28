from itertools import combinations
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


fig = plt.figure()
ax = fig.gca(projection='3d')


#dibujar cubo
r = [-1, 1]
for s, e in combinations(np.array(list(product(r,r,r))), 2):
    if np.sum(np.abs(s-e)) == r[1]-r[0]:
        ax.plot3D(*zip(s,e), color="b")

#dibujar punto
ax.scatter([0],[0],[0],color="g",s=100)


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)
print ("ingrese coordenada inicial")
#m=float(raw_input())
a = Arrow3D([0,0],[0,1],[0,0], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
b = Arrow3D([0,-1],[0,0],[0,0], mutation_scale=20, lw=1, arrowstyle="-|>", color="r")
c = Arrow3D([0,0],[0,0],[0,1], mutation_scale=20, lw=1, arrowstyle="-|>", color="b")
d = Arrow3D([0,0],[0,0],[0,-1], mutation_scale=20, lw=1, arrowstyle="-|>", color="g")
e = Arrow3D([0,1],[0,0],[0,0], mutation_scale=20, lw=1, arrowstyle="-|>", color="c")
f = Arrow3D([0,0],[0,-0.5],[0,0], mutation_scale=20, lw=1, arrowstyle="-|>", color="m")

ax.add_artist(a)
ax.add_artist(b)
ax.add_artist(c)
ax.add_artist(d)
ax.add_artist(e)
ax.add_artist(f)
plt.show()

