import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import manifold
import os

def normalize(x):
    mean_x = np.mean(x, 0, keepdims=True)
    std_x = np.std(x, 0, keepdims=True)
    print(std_x)
    x_n = (x - mean_x) / (std_x + 1e-9)
    return x_n


def fetch(dat, i, j):
    fx = dat[i, :]
    fxy = fx[:,j]
    return fxy


def func_tsne_embedding(init, resampled, joint, path):
    index = np.random.choice(400, 40)
    samples = np.random.choice(10000, 500)
    init = normalize(init)
    resampled = normalize(resampled)
    joint = normalize(joint)

    init = fetch(init, samples, index)
    resampled = fetch(resampled, samples, index)
    joint = fetch(joint, samples, index)

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    
    n1 = init.shape[0]
    n2 = resampled.shape[0]

    x_concat = np.concatenate([init, resampled, joint], axis=0)

    print('x_concat shape = ', x_concat.shape)

    x_tsne = tsne.fit_transform(x_concat)
    print('finish tsne part')
    init_tsne, resampled_tsne, joint_tsne = x_tsne[:n1, :], x_tsne[n1:n1+n2, :], x_tsne[n1+n2:, :]

    plt.figure(figsize=(6, 6))
    plt.scatter(joint_tsne[:, 0], joint_tsne[:, 1], color='red', s=0.5)
    plt.scatter(resampled_tsne[:, 0], resampled_tsne[:, 1], color='skyblue', s=0.5)
    plt.scatter(init_tsne[:, 0], init_tsne[:, 1], color='green', s=0.5)

    plt.savefig(path)
    plt.clf()
    plt.close()


def load_weights(path):
    theta1 = np.load(os.path.join(path, 'theta1.npy'))
    theta2 = np.load(os.path.join(path, 'theta2.npy'))
    theta2_fv = np.load(os.path.join(path, 'theta2_fv.npy'))
    return theta1, theta2, theta2_fv

def load_weights2(path):
    theta1 = np.load(os.path.join(path, 'theta1.npy'))
    theta2 = np.load(os.path.join(path, 'theta2_rsp.npy'))
    theta2_n = np.load(os.path.join(path, 'theta2_nse.npy'))

    return theta1, theta2, theta2_n

def calc(inp, theta1, theta2):
    h1 = np.tanh(np.matmul(inp, (theta1)))
    h2 = np.tanh(np.matmul(h1, (theta2)))
    return np.transpose(h2)

RANGE = np.pi
inp = np.expand_dims(np.arange(400) / 400.0 * RANGE * 2 - RANGE, 1)
inp = normalize(inp)

init_theta1, init_theta2, init_theta2_fv = load_weights('save_weights/sin1d3-joint/epoch0')
final_theta1, final_theta2, final_theta2_fv = load_weights('save_weights/sin1d3-resample/epoch60')

plt.figure(figsize=(8,8))
for i in range(64):
    ax=plt.subplot(8,8,i+1)
    ax.scatter(np.squeeze(inp), final_theta2_fv[i, :], color='r', s=0.8)
    #ax.scatter(np.squeeze(final_theta1), final_theta2[:, i], color='b', s=0.8)
    ax.axis('off')
plt.show()

init_theta2_fv_calc = calc(inp, init_theta1, init_theta2)
final_theta2_fv_calc = calc(inp, final_theta1, final_theta2)

print(np.mean(np.square(init_theta2_fv - init_theta2_fv_calc)))
print(np.mean(np.square(final_theta2_fv - final_theta2_fv_calc)))



for i in range(10):
    path = 'save_weights/sin1d3-joint/epoch' + str(i * 20)
    theta1, theta2, theta2_ns = load_weights(path)
    inp = normalize(inp)
    theta2_fv = calc(inp, theta1, theta2)
    #theta2_fv_ns = calc(inp, theta1, theta2_ns)

    #ideal_fv = theta2_fv + np.random.normal(0, 14.0*0.05/np.sqrt(1000), [10000,400])
    #theta1 += np.random.normal(0, 0.1/np.sqrt(1), [1, 1000])
    func_tsne_embedding(init_theta2_fv, theta2_fv, final_theta2_fv, 'logs/sin1d3-theta2-vis/epoch{}-theta2.png'.format(i))

    plt.figure(figsize=(6,6))
    plt.hist(np.squeeze(init_theta1), bins=30, normed=True, color='green', alpha=.6)
    plt.hist(np.squeeze(final_theta1), bins=30, normed=True, color='red', alpha=.6)
    plt.hist(np.squeeze(theta1), bins=30, normed=True, color='skyblue', alpha=.6)
    plt.savefig('logs/sin1d3-theta2-vis/epoch{}-theta1.png'.format(i))
    plt.clf()
    plt.close()

