import numpy as np
import os

NLAYER = 2

def update_loss(fetch, loss, need_loss=True):
    for key in fetch:
        if ('loss' in key) or (not need_loss):
            #if not need_loss:
            #    print(key)
            if key not in loss:
                loss[key] = []
            #print(fetch[key])
            loss[key].append(fetch[key])
    #print(fetch)
    #print(loss)

def print_log(title, epoch, loss):
    spacing = 10
    print_str = '{} epoch {}   '.format(title, epoch)

    for i, (k_, v_) in enumerate(loss.items()):
        if 'loss' in k_:
            #print('key = {}'.format(k_))
            value = np.around(np.mean(v_, axis=0), decimals=6)
            print_str += (k_ + ': ').rjust(spacing) + str(value) + ', '

    print_str = print_str[:-2]
    print(print_str)


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
        kxx += np.exp(-1.0 / sigma * (-2 * xx + c(x_sqnorms) + r(x_sqnorms)))
        kxy += np.exp(-1.0 / sigma * (-2 * xy + c(x_sqnorms) + r(y_sqnorms)))
        kyy += np.exp(-1.0 / sigma * (-2 * yy + c(y_sqnorms) + r(y_sqnorms)))

    mmd = np.mean(kxx) + np.mean(kyy) - 2 * np.mean(kxy)
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
    return np.sum(np.square(x), axis, keepdims=keepdims)#)


def sample_weight(weights, n, imp_sample=True):
    kernel_0, kernel_1 = weights[0], weights[1]
    kernel_0 = np.transpose(kernel_0)
    coef = l2norm(kernel_1, keepdims=False)
    coef = coef / np.sum(coef)      # [m, 0]
    m = kernel_0.shape[0]
    if imp_sample:
        sel = np.random.choice(m, n, p=coef)
        print(coef[sel])
        print('mean weight = {}'.format(np.mean(coef)))
        samples = kernel_0[sel, :]
    else:
        if n > m:
            samples = kernel_0[np.random.choice(m, n), :]
        else:
            samples = kernel_0
    return samples
