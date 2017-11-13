import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
N = 8
y = np.zeros(N)
x1 = np.linspace(0, 10, N, endpoint=True)
p1 = plt.plot(x1, y, 'o')
ax.set_xlim([-0.5,10.5])
plt.show()
