from utils import *
import numpy as np
import datetime
import tensorflow as tf
import json, sys, os
from os import path
import time
import shutil
import matplotlib
import importlib
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import *
from scipy.stats import kde
import seaborn as sns
from sklearn import manifold
from tensorflow.python.training.moving_averages import assign_moving_average

'''
2019-10-04
CUDA_VISIBLE_DEVICES=3 python mnistml.py --num_hidden 1000,1000,1 --weight_decay 0,0,0 --regw 0.01 --save_log_dir ../../data/approximate-nn/logs/mnist-l21-3l-n1 --lr 1e-4 --decay 1.0
'''


# Parse cmdline args
parser = argparse.ArgumentParser(description='MNIST')

parser.add_argument('--x_dim', default=28*28, type=int)
parser.add_argument('--gpu', default=-1, type=int)
parser.add_argument('--num_hidden', default='100,10000,1', type=str)
parser.add_argument('--weight_decay', default='0.0,0.01,1.0', type=str)
parser.add_argument('--activation', default='tanh', type=str)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--num_epoches', default=1000, type=int)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--decay', default=0.95, type=float)
parser.add_argument('--save_log_dir', default='logs/sin1d3-1000', type=str)
parser.add_argument('--train', default='all', type=str)
parser.add_argument('--save_weight_dir', type=str, default='../../data/approximate-nn/logs/sin1d3-joint')
parser.add_argument('--regw', default=0.01, type=float)
parser.add_argument('--nclass', default=10, type=int)

args = parser.parse_args()


def get_network_params():
    num_hidden = args.num_hidden.split(',')
    num_hidden = [int(x) for x in num_hidden]

    weight_decay = args.weight_decay.split(',')
    weight_decay = [float(x) for x in weight_decay]

    act = None
    if args.activation == 'tanh':
        act = tf.nn.tanh
    elif args.activation == 'relu':
        act = tf.nn.relu
    else:
        raise NotImplemented
    activation = [act] * (len(num_hidden) - 1) + [None]
    return num_hidden, weight_decay, activation

def feed_forward(x, num_hidden, decay, activation, is_training):
    depth = len(num_hidden)
    layers = [tf.identity(x)]
    l2_loss = tf.constant(0.0)
    l2_lloss = []
    inp_dim = int(x.get_shape()[-1])
    for _ in range(depth):
        print('layer {}, num hidden = {}, activation = {}, decay = {}'.format(_, num_hidden[_], activation[_], decay[_]))
        init = tf.random_normal_initializer(mean=0,stddev=1/np.sqrt(inp_dim))
        #if _ + 1 == depth:
        #    init = tf.zeros_initializer()
        #cur_layer = tf.layers.dense(layers[-1], num_hidden[_], name='dense_' + str(_), 
        #                                activation=activation[_], use_bias=True,
        #                                kernel_initializer=init)
        with tf.variable_scope('dense_' + str(_), reuse=False):
            w = tf.get_variable(name='kernel', shape=[inp_dim, num_hidden[_]], initializer=init)
            b = tf.get_variable(name='bias', shape=[1, num_hidden[_]], initializer=tf.zeros_initializer())
            cur_layer = tf.matmul(layers[-1], w) + b
            if activation[_] is not None:
                print('use activation {}'.format(activation[_]))
                cur_layer = tf.nn.tanh(cur_layer)
        #cur_layer = cur_layer / inp_dim
        #if activation[_] is not None:
        #    cur_layer = activation[_](cur_layer)
        with tf.variable_scope('dense_' + str(_), reuse=True):
            w = tf.get_variable('kernel')           # [1, m]        
        inp_dim = num_hidden[_]
        
        #reg = tf.reduce_mean(tf.reduce_sum(tf.square(w), 0))
        reg = tf.reduce_sum(tf.square(tf.reduce_mean(tf.abs(w), 1) * inp_dim)) / (inp_dim ** 2)
        #if _ == 1:
        #    reg = tf.reduce_sum(tf.abs(w)) * decay[_] / num_hidden[_]
        #else:
        #    reg = tf.reduce_sum(tf.square(w)) * decay[_] / num_hidden[_]
        l2_lloss.append(reg)
        l2_loss += reg
        layers.append(cur_layer)
    return layers[-1], l2_loss, layers, l2_lloss


def show_variables(domain, cl):
    print('Parameters in Domain {}'.format(domain))
    for item in cl:
        print('{}: {}'.format(item.name, item.shape))


def tf_add_grad_noise(all_grads, temp, lr):
    noise_grads = []
    for g, v in all_grads:
        if g is not None and len(g.get_shape()) == 2:
            g = g + tf.sqrt(lr) * temp * tf.random_normal(shape=v.get_shape(),
            #g = g * int(v.get_shape()[0]) + tf.sqrt(lr) * temp * tf.random_normal(shape=v.get_shape(), 
                mean=0, stddev=1.0/int(v.get_shape()[0]))
        noise_grads.append((g, v))
    return noise_grads


def build_model(num_hidden, decay, activation):
    x = tf.placeholder(dtype=tf.float32, shape=[None, args.x_dim])
    y = tf.placeholder(dtype=tf.int64, shape=[None])
    onehot_y = tf.one_hot(y, args.nclass)
    is_training = tf.placeholder(dtype=tf.bool, shape=[])
    lr_decay = tf.placeholder(dtype=tf.float32, shape=[])

    with tf.variable_scope('network'):
        out, reg, layers, regs = feed_forward(x, num_hidden, decay, activation, is_training)

    #print(out.get_shape(), onehot_y.get_shape())
    log_y = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_y, logits=out, dim=1)

    acc_loss = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), y), tf.float32))
    entropy_loss = tf.reduce_mean(log_y)
    loss = entropy_loss + reg * args.regw

    all_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='network')
    show_variables('All Variables', all_weights)
    last_layer_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
        scope='network/dense_{}'.format(len(num_hidden) - 1))
    show_variables('Last Layer Variables', last_layer_weights)
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='network')
    for item in update_ops:
        print('Update {}'.format(item))

    all_op = tf.train.AdamOptimizer(args.lr * lr_decay)
    all_grads = all_op.compute_gradients(loss=loss, var_list=all_weights)

    noise_grads = tf_add_grad_noise(all_grads, 1e-3, args.lr * lr_decay)
    all_train_op = all_op.apply_gradients(grads_and_vars=noise_grads)

    lst_op = tf.train.AdamOptimizer(args.lr * lr_decay)
    lst_grads = lst_op.compute_gradients(loss=loss, var_list=last_layer_weights)
    lst_train_op = lst_op.apply_gradients(grads_and_vars=lst_grads)

    weight_dict = {}
    for item in all_weights:
        if 'kernel' in item.name:
            weight_dict[item.name] = item
    print('weights to be saved')
    print(weight_dict)

    ph = {
        'x': x,
        'y': y,
        'lr_decay': lr_decay,
        'is_training': is_training,
    }

    targets = {
        'layers': layers,
        'all':{
            'weights': all_weights,
            'train': all_train_op,
            'entropy_loss': entropy_loss,
            'acc_loss': acc_loss,
            'update': update_ops,
            'reg_loss': reg,
        },
        'lst':{
            'weights': all_weights,
            'train': lst_train_op,
            'entropy_loss': entropy_loss,
            'acc_loss': acc_loss,
            'update': update_ops,
            'reg_loss': reg
        },
        'eval':{
            'weights': weight_dict,
            'entropy_loss': entropy_loss,
            'acc_loss': acc_loss,
            'out': out,
        }
    }
    for i in range(len(num_hidden)):
        targets['all']['reg{}_loss'.format(i)] = regs[i]
        #if i > 0:
        #    targets['all']['reg{}_loss'.format(i)] /= decay[i]
    return ph, targets


num_hidden, decay, activation = get_network_params()
ph, targets = build_model(num_hidden, decay, activation)

if args.gpu > -1:
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))
else:
    sess = tf.Session()

RANGE = 2 * np.pi
reg_func = cosc

mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_images = mnist.train.images # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
test_images = mnist.test.images # Returns np.array
test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

train_images = train_images.reshape(-1, args.x_dim)
train_labels = train_labels.reshape(-1)
test_images = test_images.reshape(-1, args.x_dim)
test_labels = test_labels.reshape(-1)

# scaling
mean_x = np.mean(train_images, 0, keepdims=True)
std_x = np.std(train_images, 0, keepdims=True) + 1e-9
train_images = (train_images - mean_x) / std_x
test_images = (test_images - mean_x) / std_x

ndata_train = train_images.shape[0]
ndata_test = test_images.shape[0]

nlayers = len(num_hidden)

sess.run(tf.global_variables_initializer())


def get_norm(u, axis=1):
    return np.sqrt(np.sum(np.square(u), axis))

vis_x = test_images[:400, :]
vis_y = test_labels[:400]
tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
vis_x_tsne = tsne.fit_transform(vis_x)
vis_x_mean, vis_x_std = vis_x_tsne.mean(0), vis_x_tsne.std(0)
vis_x_norm = (vis_x_tsne - vis_x_mean) / (vis_x_std)


color_set = ['red', 'blue', 'yellow', 'green', 'gray', 'black', 'orange', 'purple', 'pink', 'skyblue']
plt.figure(figsize=(8,8))
for i in range(400):
    plt.text(vis_x_norm[i, 0], vis_x_norm[i, 1], str(vis_y[i]), color=color_set[vis_y[i]], 
        fontdict={'weight': 'bold', 'size': 7})
plt.savefig(os.path.join(args.save_log_dir, 'embed.png'))
plt.close()
plt.clf()
#[0, 1]
color_set = ['red', 'blue', 'yellow', 'green', 'gray', 'black', 'orange', 'purple', 'pink', 'skyblue']
plt.figure(figsize=(8,8))
for i in range(400):
    plt.text(vis_x_norm[i, 0], vis_x_norm[i, 1], str(vis_y[i]), color=color_set[vis_y[i]], 
        fontdict={'weight': 'bold', 'size': 7})
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.savefig(os.path.join(args.save_log_dir, 'embed.png'))
plt.close()
plt.clf()
#[0, 1]


def kernel_regression(dx, dy, dv):
    # auxiliary functions
    def grid(x_l, x_r, x_steps, y_l, y_r, y_steps):
        delta_x = (x_r - x_l) / (x_steps - 1)
        delta_y = (y_r - y_l) / (y_steps - 1)
        data = np.zeros((x_steps * y_steps, 2))
        n = 0
        for i in range(x_steps):
            for j in range(y_steps):
                data[n, :] = np.array([delta_x * i + x_l, delta_y * j + y_l])
                n += 1
        return data

    def togrid(x, nx, ny):
        data = np.zeros((nx, ny))
        n = 0
        for i in range(nx):
            for j in range(ny):
                data[j, i] = x[n]
                n += 1
        return data

    xl = -2
    xr = 2
    data = grid(xl, xr, 20, xl, xr, 20)
    sigma = (xr - xl) / 20.0

    x = np.ogrid[xl:xr:20j]
    y = np.ogrid[xl:xr:20j]
    vset = {}
    v = np.ones((20, 20)) * -1
    for i in range(len(dx)):
        cx = int((dx[i] + 2) / sigma)
        cy = int((dy[i] + 2) / sigma)
        if (cx, cy) in vset:
            vset[(cx, cy)].append(dv[i])
        else:
            vset[(cx, cy)] = [dv[i]]
    for i in range(20):
        for j in range(20):
            if (i, j) in vset:
                v[i, j] = np.mean(vset[(i, j)])
    return x, y, v

def kernel_regression(dx, dy, dv):
    # auxiliary functions
    def grid(self, x_l, x_r, x_steps, y_l, y_r, y_steps):
        delta_x = (x_r - x_l) / (x_steps - 1)
        delta_y = (y_r - y_l) / (y_steps - 1)
        data = np.zeros((x_steps * y_steps, 2))
        n = 0
        for i in range(x_steps):
            for j in range(y_steps):
                data[n, :] = np.array([delta_x * i + x_l, delta_y * j + y_l])
                n += 1
        return data

    def togrid(self, x, nx, ny):
        data = np.zeros((nx, ny))
        n = 0
        for i in range(nx):
            for j in range(ny):
                data[j, i] = x[n]
                n += 1
        return data

    xl = -2
    xr = 2
    data = grid(xl, xr, 20, xl, xr, 20)
    sigma = (xr - xl) / 20.0

    x = np.ogrid[-xl:xr:20j]
    y = np.ogrid[-xl:xr:20j]
    vset = {}
    v = np.ones((20, 20)) * -1
    for i in range(len(dx)):
        cx = int((dx[i] + 2) / sigma)
        cy = int((dy[i] + 2) / sigma)
        if (cx, cy) in vset:
            vset[(cx, cy)].append(dv[i])
        else:
            vset[(cx, cy)] = [dv[i]]
    for i in range(20):
        for j in range(20):
            if (i, j) in vset:
                v[i, j] = np.mean(vset[(i, j)])

    return x, y, v


if True:
    accs = []

    for epoch in range(args.num_epoches):
        test_info = {}
        for t in tqdm(range(ndata_test // args.batch_size)):
            batch_x = test_images[t * args.batch_size: (t + 1) * args.batch_size, :]
            batch_y = test_labels[t * args.batch_size: (t + 1) * args.batch_size]

            fetch = sess.run(targets['eval'], feed_dict={ph['is_training']: False, 
                ph['x']: batch_x, ph['y']: batch_y})
            update_loss(fetch, test_info)
            fetch = sess.run(targets['layers'], feed_dict={ph['is_training']: False, 
                ph['x']: batch_x, ph['y']: batch_y, ph['lr_decay']: args.decay**(epoch)})

        accs.append(np.mean(test_info['acc_loss']))
        print_log('Test', epoch, test_info)
        
        layers_value = sess.run(targets['layers'], feed_dict={ph['is_training']: True, ph['x']: vis_x})
        for i in range(nlayers):
            layer_i = layers_value[i]     #  [B, m]
            layer_norm = np.sqrt(np.sum(np.square(layer_i), 1))
            print('Dist of layer {} norm: mean = {}, std = {}'.format(i, np.mean(layer_norm), np.std(layer_norm)))
        
        # distribution of u
        u = sess.run(targets['eval']['weights']['network/dense_{}/kernel:0'.format(nlayers - 1)])    # [1000, 1]
        u_norm = get_norm(u, 1)
        print('Distribution of u norm: {} +/- {}'.format(np.mean(u_norm), np.std(u_norm)))
        print(len(layers_value))        
        for layer_id in range(nlayers - 1): 
            thetai = sess.run(targets['eval']['weights']['network/dense_{}/kernel:0'.format(layer_id)])
            thetan = sess.run(targets['eval']['weights']['network/dense_{}/kernel:0'.format(layer_id + 1)])
            thetan_norm = get_norm(thetan, 1)
            plt.figure(figsize=(8,8))
            lvi = layers_value[layer_id + 1]
            for i in range(100):
            #if True:
                #i = 0
                ax=plt.subplot(10,10,i+1)
                px, py, pv = kernel_regression(vis_x_norm[:, 0], vis_x_norm[:, 1], lvi[:, i])
                ax.set_title('Sample %d: %.2f, %.2f, %.2f' % (i, u_norm[i], np.mean(pv), np.std(pv)), fontsize=7)
                ax.axis('off')
            plt.savefig(os.path.join(args.save_log_dir, 'f_{}_samples_{}.png'.format(layer_id, epoch)))
            plt.close()
            plt.clf()
        
            plt.figure(figsize=(8,8))
            plt.scatter(x=get_norm(thetai, 0), y=get_norm(thetan, 1), color='red', s=0.2)
            for i in range(100):
                plt.text(get_norm(thetai, 0)[i], get_norm(thetan, 1)[i], str(i), color='skyblue', 
                     fontdict={'weight': 'bold', 'size': 7})
            plt.savefig(os.path.join(args.save_log_dir, 'theta{}_index_{}.png'.format(layer_id, epoch)))
            plt.close()
            plt.clf()
            
            thetanorm = get_norm(thetai, 0)
            print('Distribution of theta {} norm: {} +/- {}'.format(layer_id, np.mean(thetanorm), np.std(thetanorm)))

        cur_idx = np.random.permutation(ndata_train)
        train_info = {}
        for t in tqdm(range(ndata_train // args.batch_size)):
            batch_idx = cur_idx[t * args.batch_size: (t + 1) * args.batch_size]
            batch_x = train_images[batch_idx, :]
            batch_y = train_labels[batch_idx]
            mode = args.train
            ep_id = epoch
            fetch = sess.run(targets[mode], feed_dict={ph['is_training']: True, 
                ph['x']: batch_x, ph['y']: batch_y, ph['lr_decay']: args.decay**(ep_id)})
            update_loss(fetch, train_info)

            #print(fetch['rmse_loss'])

        print_log('Train', epoch, train_info)
