import numpy as np
import os
import matplotlib.pyplot as plt
from utils import *

NLAYER = 2
# compute mmd distance
BASE_DIST_DIR = '../../data/approx-nn/saved_weights/mnist-1000-wz1-fix/epoch50'
COMPARE_DIST_DIR = '../../data/approx-nn/saved_weights/mnist-1000-wz1-s5555-fix'
LOG_FILE_NAME = 'logs/cp_mmd_1w.png'

nsamples1 = 1000
nsamples2 = 1000
nsamplesg = 100000
gaussian = np.random.normal(0, 1/28.0, (nsamplesg, 28*28))

w_base = load_weights(BASE_DIST_DIR)
print(np.mean(norm(w_base[1])))
#print(np.std(w_base))
#w_cand = load_weights(os.path.join(COMPARE_DIST_DIR, 'epoch49'))
ws_base = sample_weight(w_base, nsamples1, False)
#ws_cand = sample_weight(w_cand, nsamples2, False)
#ws_optimal = find_nn(ws_base, ws_cand)
print(np.std(ws_base))
print(np.std(gaussian))

def find_nn(ws_base, cand):
    cl = []
    dd = 0.0
    for i in range(ws_base.shape[0]):
        dist = np.sum(np.square(ws_base[i:i+1,:]-cand), 1)
        #print(dist.shape)
        idx = int(np.argmin(dist))
        cl.append(cand[idx, :])
        print(idx, dist[idx], np.max(dist))
        dd += np.min(dist)       
        #if i % 20 == 0:
        #    print('find nn ' + str(i))
    print('Empirical W Distance = {}'.format(dd / ws_base.shape[0]))
    return cl

#ws_optimal = find_nn(ws_base, ws_cand)

ws_optimal = find_nn(ws_base, gaussian)
mmd = mmd_fixed(ws_base, ws_optimal)
mmd2 = mmd_fixed(ws_base, gaussian[:100, :])
print('Idea mmd = {}, {}'.format(mmd, mmd2))

cand = ['init'] + ['epoch' + str(i) for i in range(100)]
mmds = []
mmd2s = []
for t in range(len(cand)):
    w_cand = load_weights(os.path.join(COMPARE_DIST_DIR, cand[t]))
    ws_base = sample_weight(w_base, nsamples1, False)
    
    ws_cand = sample_weight(w_cand, nsamples2, False)
    
    ws_base_w = sample_weight(w_base, nsamples1, True)
    ws_cand_w = sample_weight(w_cand, nsamples2, True)
    
    mmd = mmd_fixed(ws_base, ws_cand)
    mmd2 = mmd_fixed(ws_base_w, ws_cand_w)
    mmds.append(mmd)
    mmd2s.append(mmd2)
    print(mmd, mmd2)


plt.plot(np.arange(len(cand)), mmds)
plt.plot(np.arange(len(cand)), mmd2s, color='red')
plt.savefig(LOG_FILE_NAME)
plt.close()
