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
from tensorflow.python.training.moving_averages import assign_moving_average

'''
python regression.py --activation tanh --decay 1.0 --lr 1e-4 --weight_decay 0.0,0.1 --train all
'''

# Parse cmdline args
parser = argparse.ArgumentParser(description='MNIST')

parser.add_argument('--x_dim', default=1, type=int)
parser.add_argument('--gpu', default=-1, type=int)
parser.add_argument('--num_hidden', default='1000,1', type=str)
parser.add_argument('--weight_decay', default='0.0,0.0', type=str)
parser.add_argument('--activation', default='tanh', type=str)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--num_epoches', default=1000, type=int)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--decay', default=0.95, type=float)
parser.add_argument('--save_log_dir', default='logs/sin1d-1000', type=str)
parser.add_argument('--train', default='all', type=str)

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

def batch_norm(x, train, eps=1e-9, decay=0.9, affine=False, name=None):
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
        mean = tf.stop_gradient(mean)
        variance = tf.stop_gradient(variance)
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
    l2_loss = tf.constant(0.0)

    inp_dim = int(x.get_shape()[-1])
    for _ in range(depth):
        print('layer {}, num hidden = {}, activation = {}, decay = {}'.format(_, num_hidden[_], activation[_], decay[_]))
        cur_layer = tf.layers.dense(layers[-1], num_hidden[_], name='dense_' + str(_), 
                                        activation=activation[_], use_bias=False,
                                        kernel_initializer=tf.variance_scaling_initializer())
        #cur_layer = cur_layer / inp_dim
        #inp_dim = num_hidden[_]
        #with tf.variable_scope('dense_' + str(_), reuse=False):
        #    b = tf.get_variable('bias', shape=[1, num_hidden[_]])
        #    cur_layer = cur_layer + b

        with tf.variable_scope('dense_' + str(_), reuse=True):
            w = tf.get_variable('kernel')           # [1, m]        
        #if _ + 1 < depth:
        #    cur_layer = batch_norm(cur_layer, is_training)
        l2_loss += tf.reduce_sum(tf.square(w)) * decay[_] / num_hidden[_]
        layers.append(cur_layer)
    return layers[-1], l2_loss, layers


def show_variables(domain, cl):
    print('Parameters in Domain {}'.format(domain))
    for item in cl:
        print('{}: {}'.format(item.name, item.shape))


def build_model(num_hidden, decay, activation):
    x = tf.placeholder(dtype=tf.float32, shape=[None, args.x_dim])
    y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    is_training = tf.placeholder(dtype=tf.bool, shape=[])
    lr_decay = tf.placeholder(dtype=tf.float32, shape=[])

    with tf.variable_scope('network'):
        out, reg, layers = feed_forward(x, num_hidden, decay, activation, is_training)

    rmse_loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - out), 1))
    loss = rmse_loss + reg

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

    noise_grads = []
    temp = 1e-4
    for g, v in all_grads:
        if g is not None:
            g = g + tf.sqrt(args.lr * lr_decay) * temp * tf.random_normal(shape=v.get_shape(), mean=0, stddev=1)
        noise_grads.append((g, v))
    all_train_op = all_op.apply_gradients(grads_and_vars=noise_grads)

    lst_op = tf.train.AdamOptimizer(args.lr * lr_decay)
    lst_grads = lst_op.compute_gradients(loss=loss, var_list=last_layer_weights)
    lst_train_op = lst_op.apply_gradients(grads_and_vars=lst_grads)

    grad_k0, grad_k1 = None, None
    for g, v in all_grads:
        if v.name == 'network/dense_0/kernel:0':
            grad_k0 = g
        elif v.name == 'network/dense_1/kernel:0':
            grad_k1 = g
        elif v.name == 'network/dense_0/bias:0':
            grad_b0 = g
        elif v.name == 'network/dense_1/kernel:0':
            grad_b1 = g

    #lst_op = tf.train.GradientDescentOptimizer(args.lr * lr_decay)
    #lst_grads = lst_op.compute_gradients(loss=loss, var_list=last_layer_weights)
    #lst_train_op = lst_op.apply_gradients(grads_and_vars=lst_grads)

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

    targets = {
        'layers': layers,
        'all':{
            'weights': all_weights,
            'train': all_train_op,
            'rmse_loss': rmse_loss,
            'update': update_ops,
            'reg_loss': reg,
            'grads': [grad_k0, grad_k1],
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
            'out': out,
            'grads': [grad_k0, grad_k1]
        }
    }

    return ph, targets


num_hidden, decay, activation = get_network_params()
M = num_hidden[-2]
ph, targets = build_model(num_hidden, decay, activation)

if args.gpu > -1:
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))
else:
    sess = tf.Session()

RANGE = np.pi

ndata_train = 100000
train_x, train_y = sin1d(-RANGE, RANGE, 1.0, ndata_train)

# scaling
mean_x = np.mean(train_x, 0, keepdims=True)
std_x = np.std(train_x, 0, keepdims=True)
train_x = (train_x - mean_x) / std_x

sess.run(tf.global_variables_initializer())


if True:
    rmse = []

    for epoch in range(args.num_epoches):
        pp1 = []
        test_info = {}
        for t in tqdm(range(ndata_train // args.batch_size)):
            batch_x = train_x[t * args.batch_size: (t + 1) * args.batch_size, :]
            batch_y = train_y[t * args.batch_size: (t + 1) * args.batch_size]

            fetch = sess.run(targets['eval'], feed_dict={ph['is_training']: False, 
                ph['x']: batch_x, ph['y']: batch_y})
            update_loss(fetch, test_info)
            fetch = sess.run(targets['layers'], feed_dict={ph['is_training']: False, 
                ph['x']: batch_x, ph['y']: batch_y, ph['lr_decay']: args.decay**(epoch)})
            pp1.append(fetch[1])

        rmse.append(np.mean(test_info['rmse_loss']))

        pp1 = np.std(np.concatenate(pp1, 0), 0)
        #print(np.max(np.std(pp1, 0)), np.min(np.std(pp1, 0)))
        print('Std [{}, {}]: {} +/- {}'.format(np.min(pp1), np.max(pp1), np.mean(pp1), np.std(pp1)))
        print_log('Test', epoch, test_info)
        
        xp = np.arange(1000) / 1000.0 * RANGE * 2 - RANGE
        xp = np.expand_dims(xp, 1)
        xx = (xp - mean_x) / std_x
        yy = np.sin(xp)
        out = sess.run(targets['eval']['out'], feed_dict={ph['is_training']: False, 
            ph['x']: xx})

        grad_w, grad_u = sess.run(targets['all']['grads'], feed_dict={ph['is_training']: True, 
                                  ph['x']: xx, ph['y']: yy})
        grad_w = np.squeeze(grad_w)
        grad_u = np.squeeze(grad_u)

        layers_value = sess.run(targets['layers'], feed_dict={ph['is_training']: True, ph['x']: xx})
        for i in range(3):
            layer_i = layers_value[i]     #  [B, m]
            layer_norm = np.sqrt(np.sum(np.square(layer_i), 1))
            print('Dist of layer {} norm: mean = {}, std = {}'.format(i, np.mean(layer_norm), np.std(layer_norm)))

        print('Dist of u_grad: mean = {}, std = {}'.format(np.mean(np.abs(grad_u)), np.std(np.abs(grad_u))))
        print('Dist of w_grad: mean = {}, std = {}'.format(np.mean(np.abs(grad_w)), np.std(np.abs(grad_w))))        

        plt.figure(figsize=(6,6))
        plt.plot(xp, yy, color="red")
        plt.plot(xp, out, color="blue")
        plt.xlim(-RANGE, RANGE)
        plt.ylim(-1.2,1.2)
        plt.savefig(os.path.join(args.save_log_dir, 'pred_{}.png'.format(epoch)))
        plt.close()
        plt.clf()

        # distribution of u
        u = sess.run(targets['eval']['weights']['network/dense_1/kernel:0'])    # [100, 1]
        #u = pp1 / 33.0
        #u = np.sqrt(np.sum(np.square(u), 1))

        # distribution of theta
        w = sess.run(targets['eval']['weights']['network/dense_0/kernel:0'])    # [100, 10]
        w_rsp = np.squeeze(sample_weight([w, u], num_hidden[0], imp_sample=True))
        u = np.squeeze(u) #* pp1
        u_rsp = np.sum(u * (u >= 0)) / np.sum(u >= 0) * (u >= 0) + np.sum(u * (u < 0)) / np.sum(u < 0) * (u < 0)
        w = np.squeeze(w)

        plt.figure(figsize=(6,6))
        for i in range(u.shape[0]):
            plt.arrow(w[i], u[i] * np.sqrt(M), -grad_w[i] * 1e-2, -grad_u[i] * 1e-2 * np.sqrt(M), 
                length_includes_head=True, head_width=0.05, head_length=0.01)
        plt.scatter(w, u * np.sqrt(M), color='red', s=0.3)
        print(np.max(np.abs(u * np.sqrt(M))))
        plt.savefig(os.path.join(args.save_log_dir, 'grad_{}.png'.format(epoch)))
        plt.close()
        plt.clf()

        print('Dist of u: mean = {}, std = {}'.format(np.mean(u), np.std(u)))
        print('Dist of w: mean = {}, std = {}'.format(np.mean(w), np.std(w)))        

        # joint distribution
        plt.figure(figsize=(6,6))
        origin = sns.jointplot(x=w, y=u, kind='scatter', color='red')
        origin.savefig(os.path.join(args.save_log_dir, "uw_origin_{}.png".format(epoch)))
        plt.close()
        plt.clf()

        plt.figure(figsize=(6,6))
        resample = sns.jointplot(x=w_rsp, y=u_rsp, kind='scatter', color='skyblue')
        resample.savefig(os.path.join(args.save_log_dir, "uw_resample_{}.png".format(epoch)))
        resample.fig.set_size_inches(6,6)
        plt.close()
        plt.clf()

        cur_idx = np.random.permutation(ndata_train)
        train_info = {}
        for t in tqdm(range(ndata_train // args.batch_size)):
            batch_idx = cur_idx[t * args.batch_size: (t + 1) * args.batch_size]
            batch_x = train_x[batch_idx, :]
            batch_y = train_y[batch_idx]

            fetch = sess.run(targets[args.train], feed_dict={ph['is_training']: True, 
                ph['x']: batch_x, ph['y']: batch_y, ph['lr_decay']: args.decay**(epoch)})
            update_loss(fetch, train_info)
            #print(fetch['rmse_loss'])

        print_log('Train', epoch, train_info)
