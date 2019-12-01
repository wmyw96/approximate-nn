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
parser.add_argument('--regw', default=0.01, type=float)
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
    l2_loss_rsp = tf.constant(0.0)
    l2_lloss_rsp = []
    ip_weights = []
    pre_pnorm = tf.ones([inp_dim, ])
    for _ in range(depth):
        print('layer {}, num hidden = {}, activation = {}, decay = {}'.format(_, num_hidden[_], activation[_], decay[_]))
        init = tf.random_normal_initializer(mean=0,stddev=1/np.sqrt(inp_dim))
        with tf.variable_scope('nn/dense_' + str(_), reuse=False):
            w = tf.get_variable(name='kernel', shape=[inp_dim, num_hidden[_]], initializer=init)
            b = tf.get_variable(name='bias', shape=[1, num_hidden[_]], initializer=tf.zeros_initializer())
            fc = tf.matmul(layers[-1], w) + b
            cur_layer = tf.identity(fc)
            pre_layers.append(fc)
            if activation[_] is not None:
                print('use activation {}'.format(activation[_]))
                cur_layer = tf.nn.tanh(cur_layer)
        reg = decay[_] * tf.reduce_mean(tf.square(tf.reduce_mean(tf.abs(w), 1) * inp_dim)) / (inp_dim)
        with tf.variable_scope('ip', reuse=False):
            with tf.variable_scope('imp_weight_' + str(_), reuse=False):
                p_weight = tf.get_variable(name='ipweight', shape=[num_hidden[_]], initializer=tf.ones_initializer())
                p_norm = p_weight
        ip_weights.append(p_norm)
        p_norm_ = tf.expand_dims(p_norm, 0)
        reg_resample = \
            decay[_] * tf.reduce_sum(tf.square(tf.reduce_mean(tf.abs(w) * p_norm_, 1) * inp_dim) / pre_pnorm) / (inp_dim ** 2)
        l2_lloss.append(reg)
        l2_loss += reg

        pre_pnorm = p_norm

        l2_lloss_rsp.append(reg_resample)
        l2_loss_rsp += reg_resample
        inp_dim = num_hidden[_]
        #l2_lloss.append(reg)
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
        with tf.variable_scope('nn/dense_' + str(depth - _ - 1), reuse=True):
            w = tf.get_variable('kernel')
        pp = fp * tf.matmul(grad, w, transpose_b=True) * inp_dim
        pn = tf.reduce_sum(grad * fc, 1, keep_dims=True)
        print('pp shape = {}, pn shape = {}'.format(pp.get_shape(), pn.get_shape()))
        vr = tf.abs(pp - pn) #) #* np.sqrt(inp_dim)
        if _ == 0:
            p1 = tf.expand_dims(fp, 2) # [B, M, 1]
            p1 = p1 * tf.expand_dims(w, 0) * inp_dim # [B, M, k]
            p2 = tf.expand_dims(fc, 1)     # [B, 1, k]
            vr = tf.sqrt(tf.reduce_sum(tf.square(p1 - p2), 2))  # [B, M]
        eva_c = tf.reduce_mean(tf.square(vr)) / inp_dim
        eva += eva_c * (_ + 1 < depth)
        grads_norm = [tf.reduce_mean(tf.abs(grad))] + grads_norm
        grad = tf.matmul(grad, w, transpose_b=True) #* np.sqrt((inp_dim + 0.0) / out_dim)


    return layers[-1], l2_loss, layers, l2_lloss, l2_loss_rsp, l2_lloss_rsp, ip_weights, eva
       

def show_variables(domain, cl):
    print('Parameters in Domain {}'.format(domain))
    for item in cl:
        print('{}: {}'.format(item.name, item.shape))


def tf_add_grad_noise(all_grads, temp, lr):
    noise_grads = []
    for g, v in all_grads:
        if g is not None and len(g.get_shape()) == 2:
            g = g + tf.sqrt(lr) * temp * tf.random_normal(shape=v.get_shape(),
                mean=0, stddev=1.0/int(v.get_shape()[0]))
        noise_grads.append((g, v))
    return noise_grads


def normalize_grads(all_grads):
    normalized_grads = []
    for g, v in all_grads:
        if g is not None:
            g = g - tf.reduce_mean(g)
        normalized_grads.append((g, v))
    return normalized_grads


def boundary_check(x, thr=1e-7):
    ops = []
    for pweight in x:
        posind = tf.cast(tf.greater(pweight, thr), dtype=tf.float32)
        npositive = tf.reduce_sum(posind)
        positive = posind * pweight
        negative = (1.0 - posind) * thr
        cm = (tf.reduce_sum(positive) - int(pweight.get_shape()[0]) + tf.reduce_sum(negative)) / npositive
        ops.append(tf.assign(pweight, positive - posind * cm + negative))
    return ops


def build_model(num_hidden, decay, activation):
    x = tf.placeholder(dtype=tf.float32, shape=[None, args.x_dim])
    y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    layer2_copy = tf.placeholder(dtype=tf.float32, shape=[None, num_hidden[1]])
    is_training = tf.placeholder(dtype=tf.bool, shape=[])
    lr_decay = tf.placeholder(dtype=tf.float32, shape=[])

    with tf.variable_scope('network'):
        out, reg, layers, regs, reg_rsp, regs_rsp, ip_ws, eva_loss = \
            feed_forward(x, num_hidden, decay, activation, is_training)

    rmse_loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - out), 1))
    loss = rmse_loss + reg * args.regw

    rsp_loss = reg_rsp

    all_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='network/nn')
    ip_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='network/ip')

    show_variables('All Variables', all_weights)
    show_variables('Importance Weight Variables', ip_weights)

    all_op = tf.train.AdamOptimizer(args.lr * lr_decay)
    all_grads = all_op.compute_gradients(loss=loss, var_list=all_weights)
    noise_grads = tf_add_grad_noise(all_grads, 1e-1, args.lr * lr_decay)
    all_train_op = all_op.apply_gradients(grads_and_vars=noise_grads)

    rsp_op = tf.train.GradientDescentOptimizer(1e-2 * lr_decay)
    rsp_grads = rsp_op.compute_gradients(loss=rsp_loss, var_list=[ip_weights[-2]])
    rsp_grads = normalize_grads(rsp_grads)
    show_variables('Resample opt', [ip_weights[-2]])
    rsp_train_op = rsp_op.apply_gradients(grads_and_vars=rsp_grads)
    rsp_boundary = boundary_check(ip_weights)
    #rsp_train_op = 
    rsp_clear = tf.variables_initializer(ip_weights)
    #show_variables('Resample Variables', rsp_op.variables())

    all_clear = tf.variables_initializer(all_op.variables())

    weight_dict = {}
    for item in all_weights:
        if 'kernel' in item.name:
            weight_dict[item.name] = item
        if 'bias' in item.name:
            weight_dict[item.name] = item
    print(weight_dict)

    ph = {
        'x': x,
        'y': y,
        'lr_decay': lr_decay,
        'is_training': is_training,
        'layer2_copy': layer2_copy
    }
    assign = {}
    for i in range(len(num_hidden)):
        kernel_var = weight_dict['network/nn/dense_{}/kernel:0'.format(i)]
        bias_var = weight_dict['network/nn/dense_{}/bias:0'.format(i)]
        ph['dense_{}_kernel'.format(i)] = tf.placeholder(dtype=tf.float32, 
            shape=kernel_var.get_shape())
        ph['dense_{}_bias'.format(i)] = tf.placeholder(dtype=tf.float32, 
            shape=bias_var.get_shape())
        assign['dense_{}_kernel'.format(i)] = tf.assign(kernel_var, ph['dense_{}_kernel'.format(i)])
        assign['dense_{}_bias'.format(i)] = tf.assign(bias_var, ph['dense_{}_bias'.format(i)])

    targets = {
        'layers': layers,
        'all':{
            'weights': all_weights,
            'train': all_train_op,
            #'rsp_train': rsp_train_op,
            'rsp_loss': rsp_loss,
            'rmse_loss': rmse_loss,
            'reg_loss': reg,
            'eva_loss': eva_loss,
        },
        'eval':{
            'weights': weight_dict,
            'rmse_loss': rmse_loss,
            'reg_loss': reg,
            'out': out,
            'ip_weights': ip_ws,
            'ip_grad': rsp_grads,
            'eva_loss': eva_loss
        },
        'assign': assign,
        'clear': rsp_clear,
        'all_clear': all_clear,
        'rsp':{
            'rsp_train': rsp_train_op,
            'rsp_loss': rsp_loss,
        },
        'boundary': rsp_boundary,
    }
    for i in range(len(num_hidden)):
        targets['all']['reg{}_loss'.format(i)] = regs[i]
        targets['all']['reg{}_loss_rsp'.format(i)] = regs_rsp[i]
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

ndata_train = 60000
train_x, train_y = cosc_data(-RANGE, RANGE, 1.0, ndata_train)
nlayers = len(num_hidden)

# scaling
mean_x = np.mean(train_x, 0, keepdims=True)
std_x = np.std(train_x, 0, keepdims=True)
train_x = (train_x - mean_x) / std_x

sess.run(tf.global_variables_initializer())

f = open(args.logfile, 'w')
f.truncate()
f.close()

def get_norm(u, axis=1):
    return np.sqrt(np.sum(np.square(u), axis))


def get_l1norm(u, axis=1):
    return np.mean(np.abs(u), axis) * np.sqrt(u.shape[0])


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


def resample_nn(sess, ph, targets, nlayers):
    weights = sess.run(targets['eval']['weights'])
    ip = sess.run(targets['eval']['ip_weights'])
    rsp_loss, reg_loss = sess.run([targets['all']['rsp_loss'], targets['all']['reg_loss']])
    print('BEFORE: RSP LOSS = {}, REG_LOSS = {}'.format(rsp_loss, reg_loss))
    rsp0_loss, reg0_loss = sess.run([targets['all']['reg0_loss_rsp'], targets['all']['reg0_loss']])
    rsp1_loss, reg1_loss = sess.run([targets['all']['reg1_loss_rsp'], targets['all']['reg1_loss']])
    rsp2_loss, reg2_loss = sess.run([targets['all']['reg2_loss_rsp'], targets['all']['reg2_loss']])
    rsp3_loss, reg3_loss = sess.run([targets['all']['reg3_loss_rsp'], targets['all']['reg3_loss']])
    print('BEFORE: Layer 0 {}, {}; Layer 1 {}, {}; Layer 2 {}, {}; Layer 3 {}, {}'.\
        format(rsp0_loss, reg0_loss, rsp1_loss, reg1_loss, rsp2_loss, reg2_loss, rsp3_loss, reg3_loss))

    kernels = []
    biases = []
    for i in range(nlayers):
        kernels.append(weights['network/nn/dense_{}/kernel:0'.format(i)])
        biases.append(weights['network/nn/dense_{}/bias:0'.format(i)])
    #for ii in range(nlayers - 1):
    if True:
        ii = 0
        i = nlayers - 2 - ii
        curlayer = np.concatenate([kernels[i], biases[i]], 0)
        print('IP weight sum in layer {}: {}'.format(i, np.sum(ip[i])))
        curlayer_n, kernels[i + 1] = resample(curlayer, kernels[i + 1], ip[i] / curlayer.shape[1])
        kernels[i] = curlayer_n[:-1, :]
        biases[i] = np.expand_dims(curlayer_n[-1, :], 0)
    feed = {}
    for i in range(nlayers):
    #if True:
    #    i = 0
        feed[ph['dense_{}_kernel'.format(i)]] = add_noise(kernels[i])
        feed[ph['dense_{}_bias'.format(i)]] = biases[i]
    sess.run(targets['assign'], feed)
    sess.run(targets['clear'])
    reg_loss, reg0_loss, reg1_loss, reg2_loss, reg3_loss = sess.run([targets['all']['reg_loss'], targets['all']['reg0_loss'], 
        targets['all']['reg1_loss'], targets['all']['reg2_loss'], targets['all']['reg3_loss']])
    print('AFTER: REG_LOSS = {}'.format(reg_loss))
    print('AFTER: Layer 0 = {}, Layer 1 = {}, Layer 2 = {}, Layer 3 = {}'.format(reg0_loss, reg1_loss, reg2_loss, reg3_loss))
    ip = sess.run(targets['eval']['ip_weights'])
    #print(ip[0])
    reg_loss, reg0_loss, reg1_loss, reg2_loss, reg3_loss = sess.run([targets['all']['rsp_loss'], targets['all']['reg0_loss_rsp'],
        targets['all']['reg1_loss_rsp'], targets['all']['reg2_loss_rsp'], targets['all']['reg3_loss_rsp']])
    print('AFTER: Layer 0 = {}, Layer 1 = {}, Layer 2 = {}, Layer 3 = {}'.format(reg0_loss, reg1_loss, reg2_loss, reg3_loss))

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

    prersp = 0
    for epoch in range(args.num_epoches):
        test_info = {}
        for t in tqdm(range(ndata_train // args.batch_size)):
            batch_x = train_x[t * args.batch_size: (t + 1) * args.batch_size, :]
            batch_y = train_y[t * args.batch_size: (t + 1) * args.batch_size]

            fetch = sess.run(targets['eval'], feed_dict={ph['is_training']: False, 
                ph['x']: batch_x, ph['y']: batch_y})
            update_loss(fetch, test_info)
            fetch = sess.run(targets['layers'], feed_dict={ph['is_training']: False, 
                ph['x']: batch_x, ph['y']: batch_y, ph['lr_decay']: args.decay**(epoch)})


        need_resample = ((epoch % 100 == 0 and epoch <= 300) or (epoch % 10 == 0 and epoch <= 50)) and (epoch > 0)
        if need_resample:
            sess.run(targets['clear'])
            print('Optimizing the resample weights ...')
            for tt in range(100000):
                #if tt % 100 == 0:
                fetch = sess.run(targets['rsp']['rsp_train'], feed_dict={ph['lr_decay']: 20 * 0.99**(tt//5000)})
                pws = sess.run(targets['eval']['ip_weights'])
                print(np.mean(pws[-1]))
                while True:
                    ck = True
                    for pw in pws:
                        if np.min(pw) < 0:
                            ck = False
                            print('p[{}] = {}'.format(np.argmin(pw), np.min(pw)))
                    if not ck:
                        print('boundary check {}'.format(tt))
                        sess.run(targets['boundary'])
                    else:
                        break
                if tt % 100000 == 0:
                    print('Resample REG LOSS = {}'.format(sess.run(targets['rsp']['rsp_loss'])))
                

        rmse.append(np.mean(test_info['rmse_loss']))
        print_log('Test', epoch, test_info)
        
        xp = np.arange(400) / 400.0 * RANGE * 2 - RANGE
        xp = np.expand_dims(xp, 1)
        xx = (xp - mean_x) / std_x
        yy = reg_func(xp)
        out = sess.run(targets['eval']['out'], feed_dict={ph['is_training']: False, 
            ph['x']: xx})

        layers_value = sess.run(targets['layers'], feed_dict={ph['is_training']: True, ph['x']: xx})
        for i in range(nlayers):
            layer_i = layers_value[i]     #  [B, m]
            layer_norm = np.sqrt(np.sum(np.square(layer_i), 1))
            print('Dist of layer {} norm: mean = {}, std = {}'.format(i, np.mean(layer_norm), np.std(layer_norm)))
        
        plt.figure(figsize=(6,6))
        plt.plot(xp, yy, color="red")
        plt.plot(xp, out, color="blue")
        plt.xlim(-RANGE, RANGE)
        #plt.ylim(-1.2,1.2)
        plt.savefig(os.path.join(args.save_log_dir, 'pred_{}.png'.format(epoch)))
        plt.close()
        plt.clf()

        # distribution of u
        u = sess.run(targets['eval']['weights']['network/nn/dense_{}/kernel:0'.format(nlayers - 1)])    # [1000, 1]
        u_norm = get_norm(u, 1)
        print('Distribution of u norm: {} +/- {}'.format(np.mean(u_norm), np.std(u_norm)))
        print(len(layers_value))        
        for layer_id in range(nlayers - 1): 
            thetai = sess.run(targets['eval']['weights']['network/nn/dense_{}/kernel:0'.format(layer_id)])
            thetan = sess.run(targets['eval']['weights']['network/nn/dense_{}/kernel:0'.format(layer_id + 1)])
            thetan_norm = get_norm(thetan, 1)
            plt.figure(figsize=(8,8))
            lvi = layers_value[layer_id + 1]
            pweight = sess.run(targets['eval']['ip_weights'])[layer_id]
            
            if epoch % 10 == 0:
                for i in range(100):
                    ax=plt.subplot(10,10,i+1)
                    ax.scatter(np.squeeze(xp), lvi[:, i], color='r', s=0.2+0.7*thetan_norm[i]/np.max(thetan_norm))
                    ax.set_title('Sample %d: %.2f' % (i, pweight[i]), fontsize=5)
                    ax.axis('off')
                plt.savefig(os.path.join(args.save_log_dir, 'f_{}_samples_{}.png'.format(layer_id, epoch)))
                plt.close()
                plt.clf()
                np.save(os.path.join(args.save_log_dir, 'f_e{}_l{}.npy'.format(epoch, layer_id + 1)), lvi)
                np.save(os.path.join(args.save_log_dir, 'theta_e{}_l{}.npy'.format(epoch, layer_id)), thetai)
                np.save(os.path.join(args.save_log_dir, 'theta_e{}_l{}.npy'.format(epoch, layer_id + 1)), thetan)
                np.save(os.path.join(args.save_log_dir, 'pweight_e{}_l{}.npy'.format(epoch, layer_id + 1)), pweight)

            thetanorm = get_norm(thetai, 0)
            print('Distribution of theta {} norm: {} +/- {}'.format(layer_id, np.mean(thetanorm), np.std(thetanorm)))

            if not need_resample:
                 continue
            idx = (-pweight).argsort()[:100]
            print('Top k value: {}'.format(pweight[idx[0:10]]))
            print('Top k value: {}'.format(get_l1norm(thetan, 1)[idx[0:10]]))
            plt.figure(figsize=(8,8))
            for iii in range(100):
                i = idx[iii]
                ax=plt.subplot(10,10,iii+1)
                ax.scatter(np.squeeze(xp), lvi[:, i], color='r', s=0.2+0.7*thetan_norm[i]/np.max(thetan_norm))
                ax.set_title('%.2f, %.2f, %.2f' % (get_l1norm(thetai, 0)[i], get_l1norm(thetan, 1)[i], pweight[i]), fontsize=5)
                ax.axis('off')
            plt.savefig(os.path.join(args.save_log_dir, 'f_{}_samples_top_{}.png'.format(layer_id, epoch)))
            plt.close()
            plt.clf()

        do_rsp = False
        if epoch <= 50:
            if epoch % 10 == 0 and epoch > 0:
                resample_nn(sess, ph, targets, nlayers)
                do_rsp = True
                preresp=epoch
        else:
            if epoch % 100 == 0 and epoch <= 300:
                resample_nn(sess, ph, targets, nlayers)
                do_rsp = True
                preresp=epoch
        
        if not do_rsp:
            cur_idx = np.random.permutation(ndata_train)
            train_info = {}
            for t in tqdm(range(ndata_train // args.batch_size)):
                batch_idx = cur_idx[t * args.batch_size: (t + 1) * args.batch_size]
                batch_x = train_x[batch_idx, :]
                batch_y = train_y[batch_idx]
                mode = args.train
                ep_id = epoch - prersp
                fetch = sess.run(targets[mode], feed_dict={ph['is_training']: True, 
                    ph['x']: batch_x, ph['y']: batch_y, ph['lr_decay']: args.decay**(ep_id)})
                update_loss(fetch, train_info)
            print_log('Train', epoch, train_info)
        else:
            sess.run(targets['all_clear'])

        with open(args.logfile, 'a') as f:
            if not do_rsp:
                rmsec = np.mean(train_info['rmse_loss'])
            else:
                rmsec = np.mean(test_info['rmse_loss'])
            print_str = '{}, {}, {}, {}, {}\n'.format(epoch, np.mean(test_info['reg_loss']), np.mean(test_info['rmse_loss']), rmsec, np.mean(test_info['eva_loss']))
            f.write(print_str)
        #print_log('Train', epoch, train_info)
