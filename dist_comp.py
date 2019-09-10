import numpy as np
import os
import matplotlib.pyplot as plt
from utils import *

NLAYER = 2
# compute mmd distance
BASE_DIST_DIR = '../../data/approx-nn/saved_weights/mnist-100-w2/epoch99'
COMPARE_DIST_DIR = '../../data/approx-nn/saved_weights/mnist-1000/'
LOG_FILE_NAME = 'logs/cp_mmd_lst.png'

nsamples1 = 1000
nsamples2 = 1000
w_base = load_weights(BASE_DIST_DIR)
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
