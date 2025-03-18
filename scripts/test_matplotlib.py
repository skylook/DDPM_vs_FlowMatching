import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title(r'Plot of $y = \sin(x)$')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.grid()
plt.show()
