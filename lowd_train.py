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
plt.rcParams['figure.figsize']=6,6


# Parse cmdline args
parser = argparse.ArgumentParser(description='MNIST')

parser.add_argument('--x_dim', default=1, type=int)
parser.add_argument('--gpu', default=-1, type=int)
parser.add_argument('--num_hidden', default='1000,10', type=str)
parser.add_argument('--weight_decay', default='0.001,0.1', type=str)
parser.add_argument('--activation', default='tanh', type=str)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--num_epoches', default=1000, type=int)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--epoch_id', default=10, type=int)
parser.add_argument('--mode', default='train', type=str)
parser.add_argument('--save_weight_dir', default='saved_weights/mnist-1000', type=str)
parser.add_argument('--load_weight_dir', default='', type=str)
parser.add_argument('--decay', default=0.95, type=float)
parser.add_argument('--save_log_dir', default='logs/mnist-1000', type=str)
parser.add_argument('--seed', default=1234, type=int)
parser.add_argument('--train', default='lst', type=str)
parser.add_argument('--first_train', default=0, type=int)

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

args = parser.parse_args()

# GPU settings
if args.gpu > -1:
    print("GPU COMPATIBLE RUN...")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

SEED = args.seed
np.random.seed(SEED)
#tf.set_random_seed(SEED)

global_seed = tf.placeholder(tf.int32, shape=[])

from tensorflow.python.training.moving_averages import assign_moving_average

def batchnorm(x, train, eps=1e-9, decay=0.9, affine=False, name=None):
    with tf.variable_scope(name, default_name='batch_norm'):
        params_shape = x.get_shape()[-1:]
        print(params_shape)
        moving_mean = tf.get_variable('mean', params_shape,
                                      initializer=tf.zeros_initializer,
                                      trainable=False)
        moving_variance = tf.get_variable('variance', params_shape,
                                          initializer=tf.ones_initializer,
                                          trainable=False)

        def mean_var_with_update():
            mean, variance = tf.nn.moments(x, 0, name='moments')
            with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay),
                                          assign_moving_average(moving_variance, variance, decay)]):
                return tf.identity(mean), tf.identity(variance)
        mean, variance = tf.cond(train, mean_var_with_update, lambda: (moving_mean, moving_variance))
        if affine:
            beta = tf.get_variable('beta', params_shape,
                                   initializer=tf.zeros_initializer)
            gamma = tf.get_variable('gamma', params_shape,
                                    initializer=tf.ones_initializer)
            x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
        else:
            x = tf.nn.batch_normalization(x, mean, variance, None, None, eps)
        return x

def feed_forward(x, num_hidden, decay, activation, is_training):
    depth = len(num_hidden)
    layers = [tf.identity(x)]
    l2_loss = 0.0
    for _ in range(depth):
        print('layer {}, num hidden = {}, activation = {}, decay = {}'.format(_, num_hidden[_], activation[_], decay[_]))
        cur_layer = tf.layers.dense(layers[-1], num_hidden[_], name='dense_' + str(_), 
                                    activation=activation[_], use_bias=True,#(_ == 0),
                                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

        with tf.variable_scope('dense_' + str(_), reuse=True):
            w = tf.get_variable('kernel')           # [1, m]
        if _ + 1 < depth:
            #cur_layer = batchnorm(cur_layer, is_training)
            pass
            #print('use manual batch norm')
            #variance = 2 * (np.pi - tf.nn.tanh(w * np.pi) / w)
            #variance = tf.stop_gradient(variance)
            #std = tf.sqrt(variance)
            #cur_layer = cur_layer / std
            #cur_layer = tf.layers.batch_normalization(cur_layer, center=False, scale=False, training=is_training)

        l2_loss += tf.reduce_sum(tf.square(w)) * decay[_] / num_hidden[_]
        layers.append(cur_layer)
    return layers[-1], l2_loss, layers

def show_variables(cl):
    for item in cl:
        print('{}: {}'.format(item.name, item.shape))

def build_mnist_model(num_hidden, decay, activation):
    x = tf.placeholder(dtype=tf.float32, shape=[None, args.x_dim])
    y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    is_training = tf.placeholder(dtype=tf.bool, shape=[])
    with tf.variable_scope('network'):
        out, reg, layers = feed_forward(x, num_hidden, decay, activation, is_training)

    rmse_loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - out), 1))
    loss = rmse_loss + reg

    all_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='network')
    show_variables(all_weights)
    last_layer_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
        scope='network/dense_{}'.format(len(num_hidden) - 1))
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='network')
    for item in update_ops:
        print('Update {}'.format(item))

    lr_decay = tf.placeholder(dtype=tf.float32, shape=[])
    all_op = tf.train.GradientDescentOptimizer(args.lr * lr_decay)
    all_grads = all_op.compute_gradients(loss=loss, var_list=all_weights)
    all_train_op = all_op.apply_gradients(grads_and_vars=all_grads)

    lr = args.lr * lr_decay
    TEMPERATURE = 1e-8
    noise_train_ops = []
    for g, v in all_grads:
        if g is None:
            continue
        noise_train_ops.append(tf.assign(v, v - lr*g - tf.sqrt(lr)*TEMPERATURE*tf.random_normal(v.shape, stddev=1)))


    all_train_op_noise = tf.group(noise_train_ops)
    lst_op = tf.train.GradientDescentOptimizer(args.lr * lr_decay)
    lst_grads = lst_op.compute_gradients(loss=loss, var_list=last_layer_weights)
    lst_train_op = lst_op.apply_gradients(grads_and_vars=lst_grads)
    reset_lst_op = tf.variables_initializer(lst_op.variables())
    reset_all_op = tf.variables_initializer(all_op.variables())
    
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
        'is_training': is_training
    }
    ph['kernel_l0'] = tf.placeholder(dtype=tf.float32, shape=weight_dict['network/dense_0/kernel:0'].get_shape())
    #ph['bias_l0'] = tf.placeholder(dtype=tf.float32, shape=weight_dict['network/dense_0/bias:0'].get_shape())

    targets = {
        'layers': layers,
        'all':{
            'weights': all_weights,
            'train': all_train_op,
            'rmse_loss': rmse_loss,
            'update': update_ops,
            'reg_loss': reg
        },
        'all_noise':{
            'weights': all_weights,
            'train': all_train_op_noise,
            'rmse_loss': rmse_loss,
            'update': update_ops,
            'reg_loss': reg
        },
        'lst':{
            'weights': all_weights,
            'train': lst_train_op,
            'rmse_loss': rmse_loss,
            'update': update_ops,       
            'reg_loss': reg
        },
        'eval':{
            'weights': weight_dict,
            'rmse_loss': rmse_loss,
            'out': out
        },
        'assign_weights':{
            'weights_l0': tf.assign(weight_dict['network/dense_0/kernel:0'], ph['kernel_l0']),
            #'bias': tf.assign(weight_dict['network/dense_0/bias:0'], ph['bias_l0']),
        },
        'reset': {
            'lst': reset_lst_op,
            'all': reset_all_op
        }
    }

    return ph, targets


import scipy.stats as stats


def resample_2layer(sess, ph, targets, wsigma, ratio=1.0):
    kernel = sess.run(targets['eval']['weights']['network/dense_0/kernel:0'])   # [784, m]
    u = sess.run(targets['eval']['weights']['network/dense_1/kernel:0'])        # [m, 10]
    u = u * np.expand_dims(wsigma, 1)
    m = wsigma.shape[0]
    kernel_0_weight = sample_weight([kernel, u], wsigma.shape[0], imp_sample=True)
    mmd_dist = mmd_fixed(kernel_0_weight, np.transpose(kernel))
    print('mmd distance after transform = {}'.format(mmd_dist))
    kernel_resp = np.transpose(kernel_0_weight)
    sess.run(targets['reset']['all'])
    sess.run(targets['reset']['lst'])
    
    kernel_new_rv = stats.truncnorm(-1, 1, loc=0, scale=1/np.sqrt(1))
    kernel_new = kernel_new_rv.rvs((1, m))
    #sess.run(targets['eval']['weights']['network/dense_0/kernel:0'])   # [784, m]
    u_new = sess.run(targets['eval']['weights']['network/dense_1/kernel:0'])
    accept = np.random.binomial(1, ratio, (1, m))
    kernel_feed = accept * kernel_resp + (1 - accept) * kernel_new
    kernel_feed += np.random.normal(0, 0.1/np.sqrt(1))
    sess.run(targets['assign_weights']['weights_l0'], feed_dict={ph['kernel_l0']: kernel_feed})

num_hidden, decay, activation = get_network_params()
ph, targets = build_mnist_model(num_hidden, decay, activation)

if args.gpu > -1:
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))
else:
    sess = tf.Session()

RANGE = np.pi

ndata_train = 100000
train_x, train_y = sin1d(-RANGE, RANGE, 1.0, ndata_train)


sess.run(tf.global_variables_initializer())

def set_weights(val, sess, ph, targets):
    kernel = np.transpose(val, [1, 0])
    sess.run(targets['assign_weights']['weights_l0'], feed_dict={ph['kernel_l0']: kernel})


# load and set weights
if len(args.load_weight_dir) > 0:
    weights = load_weights(args.load_weight_dir)
    kernel_0 = sample_weight(weights, num_hidden[0])     # [m, 728]
    set_weights(kernel_0, sess, ph, targets)


if args.mode == 'train':

    epoch_dir = os.path.join(args.save_weight_dir, 'init')
    if not os.path.exists(epoch_dir):
        os.mkdir(epoch_dir)
    for item in targets['eval']['weights']:
        name = item
        name_saved = name.replace('/', '_', 3)
        name_saved = name_saved.replace(':', '_', 1)
        val = sess.run(targets['eval']['weights'][name])
        np.save(os.path.join(epoch_dir, name_saved + '.npy'), val)
    rmse = []

    for epoch in range(args.num_epoches):
        pp1 = []
        test_info = {}
        for t in tqdm(range(ndata_train // args.batch_size)):
            batch_x = train_x[t * args.batch_size: (t + 1) * args.batch_size, :]
            batch_y = train_y[t * args.batch_size: (t + 1) * args.batch_size]
            fetch = sess.run(targets['eval'], feed_dict={ph['is_training']: False, ph['x']: batch_x, ph['y']: batch_y})
            update_loss(fetch, test_info)
            fetch = sess.run(targets['layers'], feed_dict={ph['is_training']: False, ph['x']: batch_x, ph['y']: batch_y, ph['lr_decay']: args.decay**(epoch)})
            pp1.append(fetch[1])
            #if t == 0:
            #    print(np.max(np.std(layer1, 0)), np.min(np.std(layer1, 0)))
        rmse.append(np.mean(test_info['rmse_loss']))

        pp1 = np.std(np.concatenate(pp1, 0), 0)
        #print(np.max(np.std(pp1, 0)), np.min(np.std(pp1, 0)))
        print('Std [{}, {}]: {} +/- {}'.format(np.min(pp1), np.max(pp1), np.mean(pp1), np.std(pp1)))
        print_log('Test', epoch, test_info)

        plt.figure(figsize=(6,6))
        plt.plot(np.arange(epoch + 1), np.array(rmse), color='black')
        plt.xlim(0,50)
        plt.ylim(0,0.5)
        plt.savefig(os.path.join(args.save_log_dir, 'rmse_{}.png'.format(epoch)))
        plt.close()
        plt.clf()

        xx = np.arange(1000) / 1000.0 * RANGE * 2 - RANGE
        yy = np.sin(xx)
        out = sess.run(targets['eval']['out'], feed_dict={ph['is_training']: False, 
            ph['x']: np.expand_dims(xx, 1)})

        plt.figure(figsize=(6,6))
        plt.plot(xx, yy, color="red")
        plt.plot(xx, out, color="blue")
        plt.xlim(-RANGE, RANGE)
        plt.ylim(-1.2,1.2)
        plt.savefig(os.path.join(args.save_log_dir, 'pred_{}.png'.format(epoch)))
        plt.close()
        plt.clf()

        # distribution of u
        u = sess.run(targets['eval']['weights']['network/dense_1/kernel:0'])    # [100, 1]
        #u = np.sqrt(np.sum(np.square(u), 1))

        # distribution of theta
        w = sess.run(targets['eval']['weights']['network/dense_0/kernel:0'])    # [100, 10]
        w_rsp = np.squeeze(sample_weight([w, u], u.shape[0], imp_sample=True))
        u = np.squeeze(u) * pp1
        u_rsp = np.sum(u * (u >= 0)) / np.sum(u >= 0) * (u >= 0) + np.sum(u * (u < 0)) / np.sum(u < 0) * (u < 0)
        w = np.squeeze(w)

        print('Dist of u: mean = {}, std = {}'.format(np.mean(u), np.std(u)))
        print('Dist of w: mean = {}, std = {}'.format(np.mean(w), np.std(w)))        

        # joint distribution
        plt.figure(figsize=(6,6))
        origin = sns.jointplot(x=w, y=u, kind='scatter', color='red')
        origin.fig.set_figheight(6.12)
        origin.fig.set_figwidth(5.89)
        origin.savefig(os.path.join(args.save_log_dir, "uw_{}_origin.png".format(epoch)))
        plt.close()
        plt.clf()

        plt.figure(figsize=(6,6))
        resample = sns.jointplot(x=w_rsp, y=u_rsp, kind='scatter', color='skyblue')
        resample.fig.set_figheight(6.12)
        resample.fig.set_figwidth(5.89)
        resample.savefig(os.path.join(args.save_log_dir, "uw_{}_resample.png".format(epoch)))
        resample.fig.set_size_inches(6,6)
        plt.close()
        plt.clf()

        cur_idx = np.random.permutation(ndata_train)
        train_info = {}
        for t in tqdm(range(ndata_train // args.batch_size)):
            batch_idx = cur_idx[t * args.batch_size: (t + 1) * args.batch_size]
            batch_x = train_x[batch_idx, :]
            batch_y = train_y[batch_idx]
            mode = args.train
            if epoch < 4 and args.first_train:
                mode = 'all'

            fetch = sess.run(targets[mode], feed_dict={ph['is_training']: True, 
                ph['x']: batch_x, ph['y']: batch_y, ph['lr_decay']: args.decay**(epoch)})
            update_loss(fetch, train_info)
            #print(np.mean(fetch['rmse_loss']))
            #print(fetch[''])

        print_log('Train', epoch, train_info)

        
        #if (epoch + 1) % 50 == 0:
        #    resample_2layer(sess, ph, targets, pp1)
        # save weights
        epoch_dir = os.path.join(args.save_weight_dir, 'epoch{}'.format(epoch))
        if not os.path.exists(epoch_dir):
            os.mkdir(epoch_dir)
        for item in targets['eval']['weights']:
            name = item
            name_saved = name.replace('/', '_', 3)
            name_saved = name_saved.replace(':', '_', 1)
            val = sess.run(targets['eval']['weights'][name])
            if 'dense_1' in name:
                val = val * np.expand_dims(pp1, 1) # [m, 10]
            np.save(os.path.join(epoch_dir, name_saved + '.npy'), val)
