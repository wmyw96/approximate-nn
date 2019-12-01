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
import scipy.stats as stats
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
parser.add_argument('--logfile', default='', type=str)

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
    pre_layers = [tf.identity(x)]
    l2_loss = tf.constant(0.0)
    l2_lloss = []
    inp_dim = int(x.get_shape()[-1])
    eva = 0.0
    for _ in range(depth):
        print('layer {}, num hidden = {}, activation = {}, decay = {}'.format(_, num_hidden[_], activation[_], decay[_]))
        init = tf.random_normal_initializer(mean=0,stddev=1/np.sqrt(inp_dim))
        with tf.variable_scope('dense_' + str(_), reuse=False):
            w = tf.get_variable(name='kernel', shape=[inp_dim, num_hidden[_]], initializer=init)
            b = tf.get_variable(name='bias', shape=[1, num_hidden[_]], initializer=tf.zeros_initializer())
            fc = tf.matmul(layers[-1], w) + b
            cur_layer = tf.identity(fc)
            pre_layers.append(fc)
            if activation[_] is not None:
                print('use activation {}'.format(activation[_]))
                cur_layer = tf.nn.tanh(cur_layer)
        #reg = decay[_] * tf.reduce_mean(tf.square(tf.reduce_mean(tf.abs(w), 1) * inp_dim)) / (inp_dim)
        reg = decay[_] * tf.reduce_mean(tf.reduce_sum(tf.square(w), 0))
        inp_dim = num_hidden[_]
        #l2_lloss.append(reg)
        l2_loss += reg
        layers.append(cur_layer)
    grads_norm = []
    grad = layers[-1] * 0.0 + 1
    for _ in range(depth):
        fp = layers[depth - 1 - _]
        fc = pre_layers[depth - _]
        inp_dim = int(fp.get_shape()[1])
        out_dim = int(fc.get_shape()[1])
        print('Layer {}: {}, {}'.format(depth - _, fp.get_shape(), fc.get_shape()))
        if _ > 0:
            grad = grad * (1 - tf.square(tf.nn.tanh(fc)))
        with tf.variable_scope('dense_' + str(depth - _ - 1), reuse=True):
            w = tf.get_variable('kernel')
        pp = fp * tf.matmul(grad, w, transpose_b=True) * inp_dim
        pn = tf.reduce_sum(grad * fc, 1, keep_dims=True)
        print('pp shape = {}, pn shape = {}'.format(pp.get_shape(), pn.get_shape()))
        vr = pp - pn #) #* np.sqrt(inp_dim)
        if _ == 0:
            p1 = tf.expand_dims(fp, 2) # [B, M, 1]
            p1 = p1 * tf.expand_dims(w, 0) * inp_dim # [B, M, k]
            p2 = tf.expand_dims(fc, 1)     # [B, 1, k]
            vr = tf.sqrt(tf.reduce_sum(tf.square(p1 - p2), 2))  # [B, M]
        eva_c = tf.reduce_mean(tf.square(vr)) / inp_dim
        eva += eva_c * (_ + 1 < depth)
        l2_lloss = [eva_c] + l2_lloss
        grads_norm = [tf.reduce_mean(tf.abs(grad))] + grads_norm
        grad = tf.matmul(grad, w, transpose_b=True) #* np.sqrt((inp_dim + 0.0) / out_dim)
        #l2_lloss.append(tf.reduce_mean(tf.abs(grad)) * out_dim)


    return layers[-1], l2_loss, layers, l2_lloss, eva, grads_norm


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


def build_model(num_hidden, decay, activation, xmean, xstd):
    x = tf.placeholder(dtype=tf.float32, shape=[None, args.x_dim])
    y = tf.placeholder(dtype=tf.int64, shape=[None])
    onehot_y = tf.one_hot(y, args.nclass)
    is_training = tf.placeholder(dtype=tf.bool, shape=[])
    lr_decay = tf.placeholder(dtype=tf.float32, shape=[])
    
    with tf.variable_scope('fake_image'):
        img_w = tf.get_variable(name='img', shape=[4, args.x_dim], initializer=tf.random_normal_initializer(mean=0,stddev=1))
        #img = (tf.nn.sigmoid(img_w) * 255 - tf.convert_to_tensor(xmean)) / tf.convert_to_tensor(xstd)
        img = img_w * 1 * (xstd > 1e-6)
        print('understandable {}'.format(np.mean(xstd > 1e-6)))
    with tf.variable_scope('network', reuse=False):
        out, reg, layers, regs, eva, grad_norm = feed_forward(x, num_hidden, decay, activation, is_training)
    with tf.variable_scope('network', reuse=True):
        _, _, layers_fake, _, _, _ = feed_forward(img, num_hidden, decay, activation, is_training)

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
    
    maxijtrain = {}
    maxijclear = {}
    rc_lr = 3e-4
    for i in range(len(num_hidden)):
        for j in range(min(num_hidden[i], 100)):
            op = tf.train.AdamOptimizer(rc_lr)
            act_value = tf.reduce_mean(layers_fake[i][:, j])
            regg = tf.reduce_mean(tf.square(img))
            grads = op.compute_gradients(loss=-act_value + 0.1 * regg, var_list=[img_w])
            maxijtrain[(i, j)] = [op.apply_gradients(grads_and_vars=grads), act_value]
            maxijclear[(i, j)] = tf.variables_initializer(op.variables() + [img_w])
    
    all_op = tf.train.AdamOptimizer(args.lr * lr_decay)
    all_grads = all_op.compute_gradients(loss=loss, var_list=all_weights)
    noise_grads = all_grads
    #noise_grads = tf_add_grad_noise(all_grads, 1e-3, args.lr * lr_decay)
    all_train_op = all_op.apply_gradients(grads_and_vars=noise_grads)

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
        'good': tf.placeholder(dtype=tf.float32, shape=[4, args.x_dim])
    }
    eva_loss = eva
    targets = {
        'assign_gd': tf.assign(img_w, ph['good']),
        'image': img,
        'vis_train': maxijtrain,
        'vis_clear': maxijclear,
        'layers': layers,
        'all':{
            'weights': all_weights,
            'train': all_train_op,
            'entropy_loss': entropy_loss,
            'acc_loss': acc_loss,
            'eva_loss': eva_loss,
            'update': update_ops,
            'reg_loss': reg,
        },
        'eval':{
            'eva_loss': eva_loss,
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

#from tensorflow import keras
#fashion_mnist = keras.datasets.fashion_mnist

#mnist = fashion_mnist.load_data()
#mnist = tf.contrib.learn.datasets.load_dataset("mnist")

images = np.load('../../data/embed-mini-imagenet/train_feature_norot.npy')
labels = np.load('../../data/embed-mini-imagenet/train_label_norot.npy')
n_data = images.shape[0]
dataidx = np.random.permutation(n_data)
images = images[dataidx, :]
labels = labels[dataidx]

train_images, train_labels = images[:n_data // 10 * 7, :], labels[:n_data // 10 * 7,]
test_images, test_labels = images[n_data // 10 * 7:, :], labels[n_data // 10 * 7:,]

#train_images = mnist.train.images # Returns np.array
#train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
#test_images = mnist.test.images # Returns np.array
#test_labels = np.asarray(mnist.test.labels, dtype=np.int32)
#(train_images, train_labels), (test_images, test_labels) = mnist

#train_images = train_images.reshape(-1, args.x_dim)
#train_labels = train_labels.reshape(-1)
#test_images = test_images.reshape(-1, args.x_dim)
#test_labels = test_labels.reshape(-1)

# scaling
mean_x = np.mean(train_images, 0, keepdims=True)
std_x = np.std(train_images, 0, keepdims=True) + 1e-9
train_images = (train_images - mean_x) / std_x
test_images = (test_images - mean_x) / std_x

num_hidden, decay, activation = get_network_params()
ph, targets = build_model(num_hidden, decay, activation, mean_x, std_x)

if args.gpu > -1:
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))
else:
    sess = tf.Session()

RANGE = 2 * np.pi
reg_func = cosc


ndata_train = train_images.shape[0]
ndata_test = test_images.shape[0]

nlayers = len(num_hidden)

sess.run(tf.global_variables_initializer())


def get_norm(u, axis=1):
    return np.sqrt(np.sum(np.square(u), axis))

vis_x = test_images[:1000, :]
vis_y = test_labels[:1000]

np.save(os.path.join(args.save_log_dir, 'x_samples.npy'), vis_x)
np.save(os.path.join(args.save_log_dir, 'y_samples.npy'), vis_y)
f = open(args.logfile, 'w')
f.truncate()
f.close()

def kernel_regression(dx, dy, dv):
    # auxiliary functions
    def grid(x_l, x_r, x_steps, y_l, y_r, y_steps):
        #print('grid {}, {}, {}, {}, {}, {}'.format(x_l, x_r, x_steps, y_l, y_r, y_steps))
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
            lvi = layers_value[layer_id ]

            if epoch % 50 == 0:
                lvi = np.tanh(layers_value[layer_id ])
                for i in range(25):
                    palette = plt.get_cmap('Set1')
                    ax = plt.subplot(5, 5, i+1)
                    for j in range(10):
                        vlj = lvi[vis_y == j, i] + 0.005 * np.random.normal(size=(np.sum(vis_y == j), ))
                        gkde = stats.gaussian_kde(vlj)
                        ind = np.linspace(-1, 1, 201)
                        kdepdf = gkde.evaluate(ind)
                        ax.plot(ind, kdepdf, linewidth=0.5, label=str(j), color=palette(j))
                        #ax.set_title('Sample %d: %.2f, %.2f, %.2f' % (i, u_norm[i], np.mean(pv), np.std(pv)), fontsize=7)
                        ax.axis('off')
                plt.savefig(os.path.join(args.save_log_dir, 'f_{}_samples_{}.png'.format(layer_id + 1, epoch)))
                plt.close()
                plt.clf()           
                def get_image(x):
                    return np.reshape(x, [28, 28])
                print('visualize layer {} {}'.format(layer_id, lvi.shape))
                '''
                for iii in range(25):
                    i = np.random.choice(100)
                    while np.std(np.tanh(lvi[:, i])) <= 0.1:
                        i = np.random.choice(100)
                    idx = (-lvi[:, i]).argsort()[:4]
                    print('initial = {}'.format(np.mean(lvi[idx, i])))
                    sess.run(targets['vis_clear'][(layer_id, i)])
                    sess.run(targets['assign_gd'], feed_dict={ph['good']: vis_x[idx, :]})
                    for it in range(3001 + layer_id * 2000):
                        _, v = sess.run(targets['vis_train'][(layer_id, i)])
                        imm = sess.run(targets['image'])
                        if it % 1000 == 0:
                            print('activation value = {}, deviation max = {}, std = {}'.format(v, np.max(np.abs(imm)), np.std(imm)))
                    
                    imc = sess.run(targets['image']) * std_x + mean_x
                    im0, im1, im2, im3 = get_image(imc[0,:]), get_image(imc[1,:]), get_image(imc[2,:]), get_image(imc[3,:])
                    #im0, im1, im2, im3 = get_image(vis_x[idx[0]]), get_image(vis_x[idx[1]]), get_image(vis_x[idx[2]]), get_image(vis_x[idx[3]])
                    im = np.concatenate([np.concatenate([im0, im1], 0), np.concatenate([im2, im3], 0)], 1)
                    #print()
                    ax = plt.subplot(5, 5, iii+1)
                    ax.imshow(im, cmap='gray')
                    
                plt.savefig(os.path.join(args.save_log_dir, 'f_{}_samplesx_{}.png'.format(layer_id + 1, epoch)))
                plt.close()
                plt.clf()'''
                np.save(os.path.join(args.save_log_dir, 'f_e{}_l{}.npy'.format(epoch, layer_id + 1)), lvi)
                np.save(os.path.join(args.save_log_dir, 'theta_e{}_l{}.npy'.format(epoch, layer_id)), thetai)
                np.save(os.path.join(args.save_log_dir, 'theta_e{}_l{}.npy'.format(epoch, layer_id + 1)), thetan)
 
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

        # write different logs
        with open(args.logfile, 'a') as f:
            print_str = '{}, {}, {}, {}\n'.format(epoch, np.mean(test_info['acc_loss']), np.mean(fetch['acc_loss']), np.mean(fetch['eva_loss']))
            f.write(print_str)
        print_log('Train', epoch, train_info)
