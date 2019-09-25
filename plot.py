import matplotlib.pyplot as plt
import numpy as np

m = np.array([100, 1000, 10000, 100000])
one = np.ones(4)
sqrtm = np.sqrt(m)
ratio = np.array([0.85, 0.38, 0.22, 0.14])
plt.plot(m, one, color='red')
plt.plot(m, ratio, color='blue')
plt.xscale('log')
plt.show()