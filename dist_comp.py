import numpy as np

NLAYER = 2
# compute mmd distance
BASE_DIST_DIR = '../../data/approx-nn/saved_weights/mnist-100/epoch99'
COMPARE_DIST_DIR = '../../data/approx-nn/saved_weights/mnist-100/'
LOG_FILE_NAME = 'cp_mmd.png'

def compute_mmd(x, y, sigmas):
    xx = np.matmul(x, np.transpose(x))
    yy = np.matmul(y, np.transpose(y))
    xy = np.matmul(x, np.transpose(y))
    x_sqnorms = np.diag(xx)
    y_sqnorms = np.diag(yy)
    r = lambda x: np.expand_dims(x, 0)
    c = lambda x: np.expand_dims(x, 1)

    kxx, kxy, kyy = 0, 0, 0
    for sigma in sigmas:
        kxx += tf.exp(-1.0 / sigma * (-2 * xx + c(x_sqnorms) + r(x_sqnorms)))
        kxy += tf.exp(-1.0 / sigma * (-2 * xy + c(x_sqnorms) + r(y_sqnorms)))
        kyy += tf.exp(-1.0 / sigma * (-2 * yy + c(y_sqnorms) + r(y_sqnorms)))

    mmd = np.reduce_mean(kxx) + np.reduce_mean(kyy) - 2 * np.reduce_mean(kxy)
    return mmd


def mmd_fixed(x, y):
    fixed_sigmas = [1e-2, 1e-1, 1, 5, 10, 20, 50]
    return compute_mmd(x, y, fixed_sigmas)


def load_weights(dir_path):
    epoch_dir = dir_path
    weights = []
    for i in range(NLAYER):
        kernel = np.load(os.path.join(epoch_dir, 'network_dense_{}_kernel_0.npy'.format(i)))
        weights.append(kernel)
    return weights


def l2norm(x, axis=-1, keepdims=True):
    return np.sqrt(np.sum(np.square(x), axis, keepdims=keepdims))


def sample_weight(weights, n, imp_sample=True):
    kernel_0, kernel_1 = weights[0], weights[1]
    kernel_0 = np.transpose(kernel_0)
    coef = l2norm(kernel_1, keepdims=False)
    coef = coef / np.sum(coef)      # [m, 0]
    m = kernel_0.shape[0]
    if imp_sample:
        samples = kernel_0[np.random.choice(m, n, p=coef), :]
    else:
        if n > m:
            samples = kernel_0[np.random.choice(m, n), :, :]
        else:
            samples = kernel_0
    return samples


nsamples = 1000
w_base = load_weights(BASE_DIST_DIR)
cand = ['init'] + ['epoch' + str(i) for i in range(100)]
mmds = []
for t in range(len(cand)):
    w_cand = load_weights(os.path.joint(COMPARE_DIST_DIR, cand[t]))
    ws_base = sample_weight(w_base, nsamples, False)
    ws_cand = sample_weight(w_cand, nsamples, False)
    mmd = mmd_fixed(ws_base, ws_cand)
    mmds.append(mmd)
    printf(mmd)


plt.plot(np.arange(len(cand)), mmds)
plt.savefig(LOG_FILE_NAME)
plt.close()
