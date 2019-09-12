from utils import *
import numpy as np
import scipy.stats as stats

rv = stats.truncnorm(-1, 1, loc=0, scale=1/28.0)
s1 = rv.rvs((1000, 784))
s2 = rv.rvs((1000, 784))
dist = mmd_fixed(s1, s2)
print(dist)

