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


# Parse cmdline args
parser = argparse.ArgumentParser(description='MNIST')

parser.add_argument('--x_dim', default=1, type=int)
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

    inp_dim = int(x.get_shape()[-1])
    for _ in range(depth):
        print('layer {}, num hidden = {}, activation = {}, decay = {}'.format(_, num_hidden[_], activation[_], decay[_]))
        init = tf.random_normal_initializer(mean=0,stddev=1/np.sqrt(inp_dim))
        #if _ + 1 == depth:
        #    init = tf.zeros_initializer()
        cur_layer = tf.layers.dense(layers[-1], num_hidden[_], name='dense_' + str(_), 
                                        activation=activation[_], use_bias=False,
                                        kernel_initializer=init)

        with tf.variable_scope('dense_' + str(_), reuse=True):
            w = tf.get_variable('kernel')           # [1, m]        
        inp_dim = num_hidden[_]
        
        l2_loss += tf.reduce_sum(tf.square(w)) * decay[_] / num_hidden[_]
        layers.append(cur_layer)
    return layers[-1], l2_loss, layers


def show_variables(domain, cl):
    print('Parameters in Domain {}'.format(domain))
    for item in cl:
        print('{}: {}'.format(item.name, item.shape))


def tf_add_grad_noise(all_grads, temp, lr):
    noise_grads = []
    for g, v in all_grads:
        if g is not None:
            g = g + tf.sqrt(lr) * temp * tf.random_normal(shape=v.get_shape(), 
                mean=0, stddev=1.0/int(v.get_shape()[0]))
        noise_grads.append((g, v))
    return noise_grads


def build_model(num_hidden, decay, activation):
    x = tf.placeholder(dtype=tf.float32, shape=[None, args.x_dim])
    y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    layer2_copy = tf.placeholder(dtype=tf.float32, shape=[None, num_hidden[1]])
    is_training = tf.placeholder(dtype=tf.bool, shape=[])
    lr_decay = tf.placeholder(dtype=tf.float32, shape=[])

    with tf.variable_scope('network'):
        out, reg, layers = feed_forward(x, num_hidden, decay, activation, is_training)

    layer2_l2_loss = tf.reduce_mean(tf.reduce_mean(tf.square(layers[2] - layer2_copy), 1))
    reinit_loss = layer2_l2_loss + reg

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

    noise_grads = tf_add_grad_noise(all_grads, 1e-3, args.lr * lr_decay)
    all_train_op = all_op.apply_gradients(grads_and_vars=noise_grads)

    lst_op = tf.train.AdamOptimizer(args.lr * lr_decay)
    lst_grads = lst_op.compute_gradients(loss=loss, var_list=last_layer_weights)
    lst_train_op = lst_op.apply_gradients(grads_and_vars=lst_grads)

    reinit_op = tf.train.AdamOptimizer(args.lr * lr_decay)
    reinit_grads = reinit_op.compute_gradients(loss=reinit_loss, 
        var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='network/dense_1'))
    reinit_grads = tf_add_grad_noise(reinit_grads, 1e-3, args.lr * lr_decay)
    reinit_train_op = reinit_op.apply_gradients(grads_and_vars=reinit_grads)

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
        'layer2_copy': layer2_copy
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
        'reinit':{
            'train': reinit_train_op,
            'reinit_loss': reinit_loss,
            'reg_loss': reg,
            'layer2_l2_loss': layer2_l2_loss
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

n2 = num_hidden[-2]
n1 = num_hidden[-3]

sess.run(tf.global_variables_initializer())


def get_norm(u, axis=-1):
    return np.sqrt(np.sum(np.square(u), axis))


if True:
    rmse = []

    pp1 = []
    for t in tqdm(range(ndata_train // args.batch_size)):
        batch_x = train_x[t * args.batch_size: (t + 1) * args.batch_size, :]
        batch_y = train_y[t * args.batch_size: (t + 1) * args.batch_size]

        fetch = sess.run(targets['layers'], feed_dict={ph['is_training']: False, 
            ph['x']: batch_x, ph['y']: batch_y}) #, ph['lr_decay']: args.decay**(epoch)})
        pp1.append(fetch[2])
    train_l2v = np.concatenate(pp1, 0)

    for epoch in range(200):
        cur_idx = np.random.permutation(ndata_train)
        train_info = {}
        for t in tqdm(range(ndata_train // args.batch_size)):
            batch_idx = cur_idx[t * args.batch_size: (t + 1) * args.batch_size]
            batch_x = train_x[batch_idx, :]
            batch_l2 = train_l2v[batch_idx]
            fetch = sess.run(targets['reinit'], feed_dict={ph['is_training']: True, 
                ph['x']: batch_x, ph['layer2_copy']: batch_l2, ph['lr_decay']: args.decay**(epoch)})
            update_loss(fetch, train_info)
        print_log('Reinit', epoch, train_info)
        theta2 = sess.run(targets['eval']['weights']['network/dense_1/kernel:0'])    # [1000, 1]
        print('Theta2 Norm Mean = {}'.format(np.mean(get_norm(theta2, 0))))        


    for epoch in range(args.num_epoches):
        #pp1, pp2 = [], []
        test_info = {}
        for t in tqdm(range(ndata_train // args.batch_size)):
            batch_x = train_x[t * args.batch_size: (t + 1) * args.batch_size, :]
            batch_y = train_y[t * args.batch_size: (t + 1) * args.batch_size]

            fetch = sess.run(targets['eval'], feed_dict={ph['is_training']: False, 
                ph['x']: batch_x, ph['y']: batch_y})
            update_loss(fetch, test_info)
            fetch = sess.run(targets['layers'], feed_dict={ph['is_training']: False, 
                ph['x']: batch_x, ph['y']: batch_y, ph['lr_decay']: args.decay**(epoch)})
            #pp1.append(fetch[1])
            #pp2.append(fetch[2])

        rmse.append(np.mean(test_info['rmse_loss']))

        #pp1 = np.std(np.concatenate(pp1, 0), 0)
        #pp2 = np.std(np.concatenate(pp2, 0), 0)
        #print(np.max(np.std(pp1, 0)), np.min(np.std(pp1, 0)))
        #print('Std Layer 1 [{}, {}]: {} +/- {}'.format(np.min(pp1), np.max(pp1), np.mean(pp1), np.std(pp1)))
        #print('Std Layer 2 [{}, {}]: {} +/- {}'.format(np.min(pp2), np.max(pp2), np.mean(pp2), np.std(pp2)))
        print_log('Test', epoch, test_info)
        
        xp = np.arange(400) / 400.0 * RANGE * 2 - RANGE
        xp = np.expand_dims(xp, 1)
        xx = (xp - mean_x) / std_x
        yy = np.sin(xp)
        out = sess.run(targets['eval']['out'], feed_dict={ph['is_training']: False, 
            ph['x']: xx})

        layers_value = sess.run(targets['layers'], feed_dict={ph['is_training']: True, ph['x']: xx})
        for i in range(3):
            layer_i = layers_value[i]     #  [B, m]
            layer_norm = np.sqrt(np.sum(np.square(layer_i), 1))
            print('Dist of layer {} norm: mean = {}, std = {}'.format(i, np.mean(layer_norm), np.std(layer_norm)))
        
        plt.figure(figsize=(6,6))
        plt.plot(xp, yy, color="red")
        plt.plot(xp, out, color="blue")
        plt.xlim(-RANGE, RANGE)
        plt.ylim(-1.2,1.2)
        plt.savefig(os.path.join(args.save_log_dir, 'pred_{}.png'.format(epoch)))
        plt.close()
        plt.clf()

        # distribution of u
        u = sess.run(targets['eval']['weights']['network/dense_2/kernel:0'])    # [1000, 1]
        theta2 = sess.run(targets['eval']['weights']['network/dense_1/kernel:0'])    # [1000, 1]
        theta1 = sess.run(targets['eval']['weights']['network/dense_0/kernel:0'])    # [1000, 1]
        
        if epoch % 20 == 0:
            epoch_dir = os.path.join(args.save_weight_dir, 'epoch' + str(epoch))
            if not os.path.exists(epoch_dir):
                os.mkdir(epoch_dir)
            np.save(os.path.join(epoch_dir, 'theta1.npy'), theta1)
            print('theta2_fv shape = {}'.format(np.transpose(layers_value[2]).shape))
            np.save(os.path.join(epoch_dir, 'theta2_fv.npy'), np.transpose(layers_value[2]))
            np.save(os.path.join(epoch_dir, 'theta2.npy'), theta2)
        
        print('u shape = {}'.format(u.shape))
        print('theta2 shape = {}'.format(theta2.shape))
        print('theta1 shape = {}'.format(theta1.shape))

        u_norm = get_norm(u) * np.sqrt(n2)
        theta2_norm = get_norm(theta2) * np.sqrt(n1 / n2)
        theta1 = np.squeeze(theta1)
        #u = pp1 / 33.0
        #u = np.sqrt(np.sum(np.square(u), 1))

        print('Dist of u_norm: mean = {}, std = {}'.format(np.mean(u_norm), np.std(u_norm)))
        print('Dist of theta2_norm: mean = {}, std = {}'.format(np.mean(theta2_norm), np.std(theta2_norm)))        
        print('Dist of theta1: mean = {}, std = {}'.format(np.mean(theta1), np.std(theta1)))

        print('Theta2 Norm Mean = {}'.format(np.mean(get_norm(theta2, 0))))        
        plt.figure(figsize=(12, 4))
        ax1 = plt.subplot(1, 3, 1)
        ax1.hist(u, bins=30, normed=True, color="#FF0000", alpha=.9)
        ax2 = plt.subplot(1, 3, 2)
        ax2.hist(theta2_norm, bins=30, normed=True, color="#FF0000", alpha=.9)
        ax3 = plt.subplot(1, 3, 3)
        ax3.hist(theta1, bins=30, normed=True, color="#FF0000", alpha=.9)
        plt.savefig(os.path.join(args.save_log_dir, 'marginal_{}.png'.format(epoch)))
        plt.close()
        plt.clf()

        # joint distribution
        plt.figure(figsize=(6,6))
        origin = sns.jointplot(x=get_norm(theta2, 0)*np.sqrt(n1), y=np.squeeze(u)*np.sqrt(n2), kind='scatter', color='red')
        origin.savefig(os.path.join(args.save_log_dir, "ut2n_origin_{}.png".format(epoch)))
        plt.close()
        plt.clf()

        plt.figure(figsize=(6,6))
        origin = sns.jointplot(x=theta1, y=theta2_norm, kind='scatter', color='red')
        origin.savefig(os.path.join(args.save_log_dir, "t1t2n_origin_{}.png".format(epoch)))
        plt.close()
        plt.clf()

        cur_idx = np.random.permutation(ndata_train)
        train_info = {}
        for t in tqdm(range(ndata_train // args.batch_size)):
            batch_idx = cur_idx[t * args.batch_size: (t + 1) * args.batch_size]
            batch_x = train_x[batch_idx, :]
            batch_y = train_y[batch_idx]
            mode = args.train
            ep_id = epoch
            if epoch < 50:
                mode = 'lst'
            #print('training mode = {}'.format(mode))
            else:
                ep_id -= 50
            fetch = sess.run(targets[mode], feed_dict={ph['is_training']: True, 
                ph['x']: batch_x, ph['y']: batch_y, ph['lr_decay']: args.decay**(ep_id)})
            update_loss(fetch, train_info)
            #print(fetch['rmse_loss'])

        print_log('Train', epoch, train_info)
