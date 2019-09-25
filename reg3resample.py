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
parser.add_argument('--num_hidden', default='1000,10000,1', type=str)
parser.add_argument('--weight_decay', default='0.0,0.01,1.0', type=str)
parser.add_argument('--activation', default='tanh', type=str)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--num_epoches', default=1000, type=int)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--decay', default=0.95, type=float)
parser.add_argument('--save_log_dir', default='logs/sin1d3-1000', type=str)
parser.add_argument('--train', default='all', type=str)
parser.add_argument('--save_weight_dir', type=str, default='../../data/approximate-nn/logs/sin1d3-resample')

args = parser.parse_args()


def func_tsne_embedding(init, resampled, joint, path):
    init = normalize(init)
    resampled = normalize(resampled)
    joint = normalize(joint)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    
    n1 = init.shape[0]
    n2 = resampled.shape[0]
    x_concat = np.concatenate([init, resampled, joint], axis=0)

    x_tsne = tsne.fit_transform(x)
    init_tsne, resampled_tsne, joint_tsne = x_tsne[:n1, :], x_tsne[n1:n1+n2, :], x_tsne[n1+n2:, :]

    plt.figure(figsize=(8, 8))
    plt.scatter(joint_tsne[:, 0], joint_tsne[:, 1], color='red', s=0.5)
    plt.scatter(resampled_tsne[:, 0], resampled_tsne[:, 1], color='skyblue', s=0.5)
    plt.scatter(init_tsne[:, 0], init_tsne[:, 1], color='green', s=0.5)

    plt.savefig(path)
    plt.clf()
    plt.close()
    

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


def binary_search(n_w, dsq_w):
    l = -np.min(dsq_w) + 1e-10
    r = 1e9
    while (r - l > 1e-10):
        mid = (l + r) * 0.5
        w = n_w / np.sqrt(dsq_w + mid)
        #if np.abs(np.sum(w) - 1) < 1e-5:
        #    break
        if (np.sum(w) >= 1):
            l = mid
        else:
            r = mid
    val = mid
    print('lambda = {}'.format(val))
    w = n_w / np.sqrt(dsq_w + val)#- np.min(dsq_w) + 1e-9)
    w = w / np.sum(w)
    return w


def resample(prelayer, nextlayer, sample_weight):
    # prelayer [n_{i-1}, n_i]
    # nextlayer  [n_i, n_{i+1}]
    ni = nextlayer.shape[0]
    print('PRELAYER SHAPE = {}'.format(prelayer.shape))
    print('NEXTLAYER SHAPE = {}'.format(nextlayer.shape))
    assert ni == prelayer.shape[1]
    assert ni == sample_weight.shape[0]
    sel = np.random.choice(ni, ni, p=sample_weight)

    prelayer_rsp = prelayer[:, sel]
    nextlayer_rsp = nextlayer[sel, :] / (np.expand_dims(sample_weight[sel], 1) * ni)
    return prelayer_rsp, nextlayer_rsp

def add_noise(theta, decay=1.0):
    # theta [n_{i-1}, n_i]
    n1, n2 = theta.shape[0], theta.shape[1]
    return theta + np.random.normal(0, 0.1*decay/np.sqrt(theta.shape[0]), (n1, n2))

def resample_3layer_nn(u, theta2, theta1, decay):
    # u:        [n_2, 1]
    # theta2:   [n_1, n_2]
    # theta1:   [d, n_1]
    # decay:    list consists of 3 elements - decay[i] is the weight decay of layer i
    d = theta1.shape[0]
    n1 = theta2.shape[0]
    n2 = theta2.shape[1]
    print('Resample: d = {}, n1 = {}, n2 = {}'.format(d, n1, n2))
    weight2_n = 1.0 / n2 * np.sqrt(decay[3-1]) * np.abs(np.squeeze(u)) * np.sqrt(n2)
    weight2_dsq = decay[2-1] * np.sum(np.square(theta2), 0)
    print('weight2:numerator = {} +/- {}'.format(np.mean(weight2_n), np.std(weight2_n)))
    print('weight2:dominator = {} +/- {}'.format(np.mean(np.sqrt(weight2_dsq)), np.mean(np.sqrt(weight2_dsq))))
    weight2 = binary_search(weight2_n, weight2_dsq)
    print('weight2 resample weight sum = {}'.format(np.sum(weight2)))
    theta2, u = resample(theta2, u, weight2)
    #theta2 = add_noise(theta2)
    
    weight1_n = 1.0 / n1 * np.sqrt(decay[2-1] * np.sum(np.square(theta2), 1) / n2 * n1)
    weight1_dsq = decay[1-1] * np.sum(np.square(theta1), 0)
    weight1 = binary_search(weight1_n, weight1_dsq)

    theta1, theta2 = resample(theta1, theta2, weight1)
    #theta1 = add_noise(theta1)
    #theta2 = add_noise(theta2)
    return u, theta2, theta1

def build_model(num_hidden, decay, activation):
    x = tf.placeholder(dtype=tf.float32, shape=[None, args.x_dim])
    y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    is_training = tf.placeholder(dtype=tf.bool, shape=[])
    lr_decay = tf.placeholder(dtype=tf.float32, shape=[])

    with tf.variable_scope('network'):
        out, reg, layers = feed_forward(x, num_hidden, decay, activation, is_training)

    rmse_loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - out), 1))
    loss = rmse_loss + 0.01 * reg

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
    reset_lst_op = tf.variables_initializer(lst_op.variables())
    reset_all_op = tf.variables_initializer(all_op.variables())

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
        'is_training': is_training
    }
    ph['kernel_l0'] = tf.placeholder(dtype=tf.float32, shape=weight_dict['network/dense_0/kernel:0'].get_shape())
    ph['kernel_l1'] = tf.placeholder(dtype=tf.float32, shape=weight_dict['network/dense_1/kernel:0'].get_shape())
    ph['kernel_l2'] = tf.placeholder(dtype=tf.float32, shape=weight_dict['network/dense_2/kernel:0'].get_shape())

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
        },
        'assign_weights':{
            'weights_l0': tf.assign(weight_dict['network/dense_0/kernel:0'], ph['kernel_l0']),
            'weights_l1': tf.assign(weight_dict['network/dense_1/kernel:0'], ph['kernel_l1']),
            'weights_l2': tf.assign(weight_dict['network/dense_2/kernel:0'], ph['kernel_l2']),
        },
        'reset': {
            'lst': reset_lst_op,
            'all': reset_all_op
        }
    }

    return ph, targets


num_hidden, decay, activation = get_network_params()
M = num_hidden[-2]
ph, targets = build_model(num_hidden, decay, activation)

decay_resample = []
#for i in range(len(decay)):
decay_resample = decay
    #decay_resample.append((decay[i] + 0.0) / np.sqrt(num_hidden[i]))
print('DECAY RESAMPLE = {}'.format(decay_resample))

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


def update_weights(sess, ph, targets, u, theta2, theta1):
    sess.run(targets['reset']['all'])
    sess.run(targets['reset']['lst'])
    
    sess.run(targets['assign_weights']['weights_l0'], feed_dict={ph['kernel_l0']: theta1})
    sess.run(targets['assign_weights']['weights_l1'], feed_dict={ph['kernel_l1']: theta2})
    sess.run(targets['assign_weights']['weights_l2'], feed_dict={ph['kernel_l2']: u})


if True:
    rmse = []
    noise_decay = 1.0
    for epoch in range(args.num_epoches):
        pp1, pp2 = [], []
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
            pp2.append(fetch[2])

        rmse.append(np.mean(test_info['rmse_loss']))

        pp1 = np.std(np.concatenate(pp1, 0), 0)
        pp2 = np.std(np.concatenate(pp2, 0), 0)
        #print(np.max(np.std(pp1, 0)), np.min(np.std(pp1, 0)))
        print('Std Layer 1 [{}, {}]: {} +/- {}'.format(np.min(pp1), np.max(pp1), np.mean(pp1), np.std(pp1)))
        print('Std Layer 2 [{}, {}]: {} +/- {}'.format(np.min(pp2), np.max(pp2), np.mean(pp2), np.std(pp2)))
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
        u = sess.run(targets['eval']['weights']['network/dense_2/kernel:0'])    # [n_2, 1]
        theta2 = sess.run(targets['eval']['weights']['network/dense_1/kernel:0'])    # [n_1, n_2]
        theta1 = sess.run(targets['eval']['weights']['network/dense_0/kernel:0'])    # [d, n_1]

        #if epoch % 20 == 0:
        #    epoch_dir = os.path.join(args.save_weight_dir, 'epoch' + str(epoch))
        #    if not os.path.exists(epoch_dir):
        #        os.mkdir(epoch_dir)
        #    np.save(os.path.join(epoch_dir, 'theta1.npy'), theta1)
        #    print('theta2_fv shape = {}'.format(np.transpose(layers_value[2]).shape))
        #    np.save(os.path.join(epoch_dir, 'theta2_fv.npy'), np.transpose(layers_value[2]))

        def calc(inp, theta1, theta2):
            h1 = np.tanh(np.matmul(inp, (theta1)))
            h2 = np.tanh(np.matmul(h1, (theta2)))
            return np.transpose(h2)
        inp = xx
        
        if epoch % 20 == 0:
            u_c, theta2_c, theta1_c = u, theta2, theta1
            for i in range(50):
                u_c, theta2_c, theta1_c = resample_3layer_nn(u_c, theta2_c, theta1_c, decay_resample)
                print('Iter {}: U norm = {}, Theta2 norm = {}, Theta1 norm = {}'.format(i, np.mean(u_c*u_c), np.mean(theta2_c*theta2_c), np.mean(theta1_c * theta1_c)))
                noise_decay = 0.6 * (0.98)**(epoch//5) + 0.4
                if i % 2 == 0:
                    plt.figure(figsize=(8,8))
                    final_theta2_fv = calc(inp, theta1_c, theta2_c)
                    for k in range(64):
                        ax=plt.subplot(8,8,k+1)
                        ax.scatter(np.squeeze(inp), final_theta2_fv[k, :], color='r', s=0.8)
                        #ax.scatter(np.squeeze(final_theta1), final_theta2[:, i], color='b', s=0.8)
                        ax.axis('off')
                    plt.savefig(os.path.join(args.save_log_dir, 'rsp_f2_{}_{}.png'.format(epoch, i)))
                    plt.clf()
                    plt.close()
                    plt.figure(figsize=(6,6))
                    plt.hist(np.squeeze(theta1_c), bins=30, normed=True, color='red', alpha=.6)
                    plt.savefig(os.path.join(args.save_log_dir, 'rsp_t1_{}_{}.png'.format(epoch, i)))
                    plt.clf()
                    plt.close()

                theta2_c, theta1_c = add_noise(theta2_c, noise_decay * (i + 1) / 25), add_noise(theta1_c, noise_decay * (i + 1) / 25)
            u_rsp, theta2_rsp, theta1_rsp = u_c, theta2_c, theta1_c #add_noise(theta2_c, noise_decay), add_noise(theta1_c, noise_decay) #resample_3layer_nn(u, theta2, theta1, decay)
                    # resample
            #plt.figure(figsize=(6,6))
            #origin = sns.jointplot(x=get_norm(theta2_rsp, 0)*np.sqrt(n1), y=np.squeeze(u_rsp)*np.sqrt(n2), kind='scatter', color='red')
            #origin.savefig(os.path.join(args.save_log_dir, "ut2n_resample_{}.png".format(epoch)))
            #plt.close()
            #plt.clf()

            #plt.figure(figsize=(6,6))
            #origin = sns.jointplot(x=np.squeeze(theta1_rsp), y=get_norm(theta2_rsp)*np.sqrt(n1), kind='scatter', color='red')
            #origin.savefig(os.path.join(args.save_log_dir, "t1t2n_resample_{}.png".format(epoch)))
            #plt.close()
            #plt.clf()

        if epoch % 40 == 0:
            epoch_dir = os.path.join(args.save_weight_dir, 'epoch' + str(epoch))
            if not os.path.exists(epoch_dir):
                os.mkdir(epoch_dir)
            np.save(os.path.join(epoch_dir, 'theta1.npy'), theta1)
            np.save(os.path.join(epoch_dir, 'theta2.npy'), theta2)
            print('theta2_fv shape = {}'.format(np.transpose(layers_value[2]).shape))
            np.save(os.path.join(epoch_dir, 'theta1_rsp.npy'), theta1_c)
            np.save(os.path.join(epoch_dir, 'theta2_rsp.npy'), theta2_c)
            np.save(os.path.join(epoch_dir, 'theta2_nse.npy'), theta2_rsp)

        print('u shape = {}'.format(u.shape))
        print('theta2 shape = {}'.format(theta2.shape))
        print('theta1 shape = {}'.format(theta1.shape))

        u_norm = get_norm(u) * np.sqrt(n2)
        theta2_norm = get_norm(theta2) * np.sqrt(n1/n2)
        theta1 = np.squeeze(theta1)
        #u = pp1 / 33.0
        #u = np.sqrt(np.sum(np.square(u), 1))

        print('Dist of u_norm: mean = {}, std = {}'.format(np.mean(u_norm), np.std(u_norm)))
        print('Dist of theta2_norm: mean = {}, std = {}'.format(np.mean(theta2_norm), np.std(theta2_norm)))        
        print('Dist of theta1: mean = {}, std = {}'.format(np.mean(theta1), np.std(theta1)))

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

        # resample
        #plt.figure(figsize=(6,6))
        #origin = sns.jointplot(x=get_norm(theta2_rsp, 0)*np.sqrt(n1), y=np.squeeze(u_rsp)*np.sqrt(n2), kind='scatter', color='red')
        #origin.savefig(os.path.join(args.save_log_dir, "ut2n_resample_{}.png".format(epoch)))
        #plt.close()
        #plt.clf()

        #plt.figure(figsize=(6,6))
        #origin = sns.jointplot(x=np.squeeze(theta1_rsp), y=get_norm(theta2_rsp)*np.sqrt(n1), kind='scatter', color='red')
        #origin.savefig(os.path.join(args.save_log_dir, "t1t2n_resample_{}.png".format(epoch)))
        #plt.close()
        #plt.clf()

        if (epoch + 1) % 50 == 0:
            update_weights(sess, ph, targets, u_rsp, theta2_rsp, theta1_rsp)
        else:
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
