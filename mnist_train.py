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


# Parse cmdline args
parser = argparse.ArgumentParser(description='MNIST')

parser.add_argument('--x_dim', default=28*28, type=int)
parser.add_argument('--gpu', default=-1, type=int)
parser.add_argument('--nclass', default=10, type=int)
parser.add_argument('--num_hidden', default='1000,10', type=str)
parser.add_argument('--weight_decay', default='0.001,0.1', type=str)
parser.add_argument('--activation', default='tanh', type=str)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--num_epoches', default=1000, type=int)
parser.add_argument('--batch_size', default=128, type=int)
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
tf.set_random_seed(SEED)

from tensorflow.python.training.moving_averages import assign_moving_average

def batchnorm(x, train, eps=1e-05, decay=0.9, affine=False, name=None):
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
                                    activation=activation[_], use_bias=False,#(_ == 0),
                                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        if _ + 1 < depth:
            #print('use batch norm')
            cur_layer = batchnorm(cur_layer, is_training)
            #cur_layer = tf.layers.batch_normalization(cur_layer, center=False, scale=False, training=is_training)
        with tf.variable_scope('dense_' + str(_), reuse=True):
            w = tf.get_variable('kernel')
        l2_loss += tf.reduce_sum(tf.square(w)) * decay[_] / num_hidden[_]
        layers.append(cur_layer)
    return layers[-1], l2_loss, layers

def show_variables(cl):
    for item in cl:
        print('{}: {}'.format(item.name, item.shape))

def build_mnist_model(num_hidden, decay, activation):
    x = tf.placeholder(dtype=tf.float32, shape=[None, args.x_dim])
    y = tf.placeholder(dtype=tf.int64, shape=[None])
    is_training = tf.placeholder(dtype=tf.bool, shape=[])
    onehot_y = tf.one_hot(y, args.nclass)
    with tf.variable_scope('network'):
        out, reg, layers = feed_forward(x, num_hidden, decay, activation, is_training)
    log_y = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_y, logits=out, dim=1)

    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), y), tf.float32))
    entropy_loss = tf.reduce_mean(log_y)
    loss = entropy_loss + reg

    all_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='network')
    show_variables(all_weights)
    last_layer_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
        scope='network/dense_{}'.format(len(num_hidden) - 1))
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='network')
    for item in update_ops:
        print('Update {}'.format(item))

    lr_decay = tf.placeholder(dtype=tf.float32, shape=[])
    all_op = tf.train.AdamOptimizer(args.lr * lr_decay)
    all_grads = all_op.compute_gradients(loss=loss, var_list=all_weights)
    all_train_op = all_op.apply_gradients(grads_and_vars=all_grads)
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
        'is_training': is_training
    }
    ph['kernel_l0'] = tf.placeholder(dtype=tf.float32, shape=weight_dict['network/dense_0/kernel:0'].get_shape())
    #ph['bias_l0'] = tf.placeholder(dtype=tf.float32, shape=weight_dict['network/dense_0/bias:0'].get_shape())

    targets = {
        'layers': layers,
        'all':{
            'weights': all_weights,
            'train': all_train_op,
            'acc_loss': acc,
            'entropy_loss': entropy_loss,
            'update': update_ops
        },
        'lst':{
            'weights': all_weights,
            'train': lst_train_op,
            'acc_loss': acc,
            'entropy_loss': entropy_loss,
            'update': update_ops       
        },
        'eval':{
            'weights': weight_dict,
            'acc_loss': acc,
            'entropy_loss': entropy_loss    
        },
        'assign_weights':{
            'weights_l0': tf.assign(weight_dict['network/dense_0/kernel:0'], ph['kernel_l0']),
            #'bias': tf.assign(weight_dict['network/dense_0/bias:0'], ph['bias_l0']),
        }
    }

    return ph, targets


num_hidden, decay, activation = get_network_params()
ph, targets = build_mnist_model(num_hidden, decay, activation)

if args.gpu > -1:
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))
else:
    sess = tf.Session()


mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_images = mnist.train.images # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
test_images = mnist.test.images # Returns np.array
test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

train_images = train_images.reshape(-1, args.x_dim)
train_labels = train_labels.reshape(-1)
test_images = test_images.reshape(-1, args.x_dim)
test_labels = test_labels.reshape(-1)

ndata_train = train_images.shape[0]
ndata_test = test_images.shape[0]

# do normalization
#tr_mean = np.mean(train_images, 0, keepdims=True)
#tr_std = np.std(train_images, 0, keepdims=True) + 1e-8
#print(np.min(tr_std))
#train_images = (train_images - tr_mean) / tr_std
#test_images = (test_images - tr_mean) / tr_std

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

    for epoch in range(args.num_epoches):
        # distribution of u
        u = sess.run(targets['eval']['weights']['network/dense_1/kernel:0'])    # [100, 10]
        u = np.sqrt(np.sum(np.square(u), 1))
        plt.hist(u, bins=30, normed=True, color="#FF0000", alpha=.9)
        plt.savefig(os.path.join(args.save_log_dir, 'u_dist_epoch_{}.png'.format(epoch)))
        plt.close()
        print('Dist of u: mean = {}, std = {}'.format(np.mean(u), np.std(u)))

        cur_idx = np.random.permutation(ndata_train)
        train_info = {}
        for t in tqdm(range(ndata_train // args.batch_size)):
            batch_idx = cur_idx[t * args.batch_size: (t + 1) * args.batch_size]
            batch_x = train_images[batch_idx, :]
            batch_y = train_labels[batch_idx]
            mode = args.train
            if epoch < 4 and args.first_train:
                mode = 'all'
            
            fetch = sess.run(targets['layers'], feed_dict={ph['is_training']: True, ph['x']: batch_x, ph['y']: batch_y, ph['lr_decay']: args.decay**epoch})
            layer1 = fetch[1]
            #print(layer1.shape)
            #print(np.mean(layer1, 0))
            #if t == 0:
            #    print(np.max(np.std(layer1, 0)), np.min(np.std(layer1, 0)))
            fetch = sess.run(targets[mode], feed_dict={ph['is_training']: True, ph['x']: batch_x, ph['y']: batch_y, ph['lr_decay']: args.decay**epoch})
            update_loss(fetch, train_info)
        
        pp1 = []
        test_info = {}
        for t in tqdm(range(ndata_train // args.batch_size)):
            batch_x = train_images[t * args.batch_size: (t + 1) * args.batch_size, :]
            batch_y = train_labels[t * args.batch_size: (t + 1) * args.batch_size]
            fetch = sess.run(targets['eval'], feed_dict={ph['is_training']: False, ph['x']: batch_x, ph['y']: batch_y})
            update_loss(fetch, test_info)
            fetch = sess.run(targets['layers'], feed_dict={ph['is_training']: False, ph['x']: batch_x, ph['y']: batch_y, ph['lr_decay']: args.decay**epoch})
            pp1.append(fetch[1])
            #if t == 0:
            #    print(np.max(np.std(layer1, 0)), np.min(np.std(layer1, 0)))
        pp1 = np.std(np.concatenate(pp1, 0), 0)
        #print(np.max(np.std(pp1, 0)), np.min(np.std(pp1, 0)))
        print('Std [{}, {}]: {} +/- {}'.format(np.min(pp1), np.max(pp1), np.mean(pp1), np.std(pp1)))
        print_log('Train', epoch, train_info)
        print_log('Test', epoch, test_info)

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
                val = val * np.expand_dims(pp1, 1) # [10, m]
            np.save(os.path.join(epoch_dir, name_saved + '.npy'), val)
