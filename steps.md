## Steps w.r.t two-level Neural Networks

```
python mnist_train.py --num_hidden 10000,10 --weight_decay 0.0,0.0 --lr 1e-3 --save_weight_dir ../../data/approx-nn/saved_weights/mnist-10000 --save_log_dir logs/mnist-10000
```

- default training:
    - m=1000, seed=1234: 100/98
    - 
```
python mnist_train.py --num_hidden 1000,10 --weight_decay 0.001,0.1 --lr 1e-3 --decay 1.0 --save_weight_dir ../../data/approx-nn/saved_weights/mnist-1000 --save_log_dir logs/mnist-1000 --num_epoch 100 --train all
python mnist_train.py --num_hidden 100,10 --weight_decay 0.001,0.2 --lr 1e-3 --decay 1.0 --save_weight_dir ../../data/approx-nn/saved_weights/mnist-100-w2 --save_log_dir logs/mnist-100-w2 --num_epoch 100 --train all

python mnist_train.py --num_hidden 10000,10 --weight_decay 0.0,0.0 --lr 1e-3 --decay 1.0 --save_weight_dir ../../data/approx-nn/saved_weights/mnist-10000 --save_log_dir logs/mnist-10000 --num_epoch 100 --train lst --gpu 1

python mnist_train.py --num_hidden 10,10 --weight_decay 0.0,0.0 --lr 1e-3 --decay 1.0 --save_weight_dir ../../data/approx-nn/saved_weights/mnist-10 --load_weight_dir ../../data/approx-nn/saved_weights/mnist-10000/epoch99 --save_log_dir logs/mnist-10 --num_epoch 100 --train lst --gpu 1


python mnist_train.py --num_hidden 10000,10 --weight_decay 0.0,0.0 --lr 1e-3 --decay 1.0 --save_weight_dir ../../data/approx-nn/saved_weights/mnist-10000 --save_log_dir logs/mnist-10000 --num_epoch 100 --train lst --gpu 1

```


```
sample complexity:


python mnist_train.py --num_hidden 10000,10 --weight_decay 0.001,0.1 --lr 1e-3 --decay 1.0 --save_weight_dir ../../data/approx-nn/saved_weights/mnist-10000-wz1 --save_log_dir logs/mnist-10000-wz1 --num_epoch 100 --train lst --gpu 1

python mnist_train.py --num_hidden 10000,10 --weight_decay 0.0001,0.01 --lr 1e-3 --decay 1.0 --save_weight_dir ../../data/approx-nn/saved_weights/mnist-10000-wz1 --save_log_dir logs/mnist-10000-wz1 --num_epoch 100 --train lst --gpu 1

python mnist_train.py --num_hidden 10,10 --weight_decay 0.0,0.0 --lr 1e-3 --decay 1.0 --save_weight_dir ../../data/approx-nn/saved_weights/mnist-10 --load_weight_dir ../../data/approx-nn/saved_weights/mnist-10000-wz1/epoch99 --save_log_dir logs/mnist-10 --num_epoch 100 --train lst --gpu 1

# fake
python mnist_train.py --num_hidden 10000,10 --weight_decay 0.001,1.0 --lr 1e-3 --decay 1.0 --save_weight_dir ../../data/approx-nn/saved_weights/mnist-10000-w1 --save_log_dir logs/mnist-10000-w1 --num_epoch 100 --train all --gpu 1
python mnist_train.py --num_hidden 10,10 --weight_decay 0.0,0.0 --lr 1e-3 --decay 1.0 --save_weight_dir ../../data/approx-nn/saved_weights/mnist-10 --load_weight_dir ../../data/approx-nn/saved_weights/mnist-10000-w1/epoch99 --save_log_dir logs/mnist-10 --num_epoch 100 --train lst --gpu 1

python mnist_train.py --num_hidden 10000,10 --weight_decay 0.001,10.0 --lr 1e-3 --decay 1.0 --save_weight_dir ../../data/approx-nn/saved_weights/mnist-10000-w10 --save_log_dir logs/mnist-10000-w10 --num_epoch 100 --train all --gpu 1

python mnist_train.py --num_hidden 10,10 --weight_decay 0.0,0.0 --lr 1e-3 --decay 1.0 --save_weight_dir ../../data/approx-nn/saved_weights/mnist-10 --load_weight_dir ../../data/approx-nn/saved_weights/mnist-10000-w10/epoch99 --save_log_dir logs/mnist-10 --num_epoch 100 --train lst --gpu 1


python mnist_train.py --num_hidden 10,10 --weight_decay 0.0,0.0 --lr 1e-3 --decay 1.0 --save_weight_dir ../../data/approx-nn/saved_weights/mnist-10 --load_weight_dir ../../data/approx-nn/saved_weights/mnist-10000-w10/epoch99 --save_log_dir logs/mnist-10 --num_epoch 100 --train lst --gpu 1

python mnist_train.py --num_hidden 10,10 --weight_decay 0.0,0.0 --lr 1e-3 --decay 1.0 --save_weight_dir ../../data/approx-nn/saved_weights/mnist-10 --load_weight_dir ../../data/approx-nn/saved_weights/mnist-10000-w10-fix/epoch99 --save_log_dir logs/mnist-10 --num_epoch 100 --train lst --gpu 1



python mnist_train.py --num_hidden 10000,10 --weight_decay 0.001,0.1 --lr 1e-3 --decay 1.0 --save_weight_dir ../../data/approx-nn/saved_weights/mnist-10000-wz1-fix --save_log_dir logs/mnist-10000-wz1-fix --num_epoch 100 --train all --gpu 1

# last 
python mnist_train.py --num_hidden 10000,10 --weight_decay 0.0,0.0 --lr 1e-3 --decay 1.0 --save_weight_dir ../../data/approx-nn/saved_weights/mnist-10000-w0-lst-fix --save_log_dir logs/mnist-10000-w0-lst-fix --num_epoch 400 --train lst --gpu 1
```

- could two-level nn with random first layer learn competitive results compared with original neural networks
    - m=10000: 98/96
    - m=1000: 97/95

```
python mnist_train.py --num_hidden 1000,10 --weight_decay 0.0,0.0 --lr 1e-3 --decay 1.0 --save_weight_dir ../../data/approx-nn/saved_weights/mnist-1000-lst --save_log_dir logs/mnist-1000-lst --num_epoch 500 --train lst


```

- could the distribution of two overaparmaterized neural network be similar?
    - 

### Sample Complexity

- base setting: m=10000, lr=1e-3, no decay
- full training, cw=1e-3, cu=10.0 (same with Dong): accuracy=82%
```
python mnist_train.py --num_hidden 10000,10 --weight_decay 0.001,10.0 --lr 1e-3 --decay 1.0 --save_weight_dir ../../data/approx-nn/saved_weights/mnist-10000-w10-fix --save_log_dir logs/mnist-10000-w10-fix --num_epoch 100 --train all --gpu 1

python mnist_train.py --num_hidden 10,10 --weight_decay 0.0,0.0 --lr 1e-3 --decay 1.0 --save_weight_dir ../../data/approx-nn/saved_weights/mnist-10 --load_weight_dir ../../data/approx-nn/saved_weights/mnist-10000-w10-fix/epoch20 --save_log_dir logs/mnist-10 --num_epoch 100 --train lst --gpu 2

```
- full training, cw=1e-3, cu=0.1 (normal concentrate case): accuracy=74% (=77 at epoch 3)
```
python mnist_train.py --num_hidden 10000,10 --weight_decay 0.001,1.0 --lr 1e-3 --decay 1.0 --save_weight_dir ../../data/approx-nn/saved_weights/mnist-10000-w1-fix --save_log_dir logs/mnist-10000-w1-fix --num_epoch 100 --train all --gpu 1

python mnist_train.py --num_hidden 10,10 --weight_decay 0.0,0.0 --lr 1e-3 --decay 1.0 --save_weight_dir ../../data/approx-nn/saved_weights/mnist-10 --load_weight_dir ../../data/approx-nn/saved_weights/mnist-10000-wz1-fix/epoch3 --save_log_dir logs/mnist-10 --num_epoch 100 --train lst --gpu 1
```

- lst training: accuracy=53%
```
python mnist_train.py --num_hidden 10,10 --weight_decay 0.0,0.0 --lr 1e-3 --decay 1.0  --save_weight_dir ../../data/approx-nn/saved_weights/mnist-10 --load_weight_dir ../../data/approx-nn/saved_weights/mnist-10000-w0-lst-fix/epoch99 --save_log_dir logs/mnist-10 --num_epoch 100 --train lst --gpu 1

```

- lst training + first 4 full training, cw=1e-3, cu=0.1: accuracy=75% (training accuracy coun't exteed to 96%)
```
python mnist_train.py --num_hidden 10000,10 --weight_decay 0.001,0.1 --lr 1e-3 --decay 1.0 --save_weight_dir ../../data/approx-nn/saved_weights/mnist-10000-lst-fix --save_log_dir logs/mnist-10000-lst-fix --num_epoch 400 --train lst --gpu 1 

python mnist_train.py --num_hidden 10,10 --weight_decay 0.0,0.0 --lr 1e-3 --decay 1.0 --save_weight_dir ../../data/approx-nn/saved_weights/mnist-10 --load_weight_dir ../../data/approx-nn/saved_weights/mnist-10000-f-lst-fix/epoch99 --save_log_dir logs/mnist-10 --num_epoch 100 --train lst --gpu 1

```



```
python mnist_train.py --num_hidden 10000,10 --weight_decay 0.0001,1.0 --lr 1e-3 --decay 1.0 --save_weight_dir ../../data/approx-nn/saved_weights/mnist-1000-wz1-fix --save_log_dir logs/mnist-10000-w1-fix --num_epoch 100 --train all --gpu 1


python mnist_train.py --num_hidden 1000,10 --weight_decay 0.001,0.1 --lr 1e-3 --decay 1.0 --save_weight_dir ../../data/approx-nn/saved_weights/mnist-1000-wz1-lst-fix --save_log_dir logs/mnist-1000-wz1-lst-fix --num_epoch 100 --train lst --gpu 1 --first_train 1

python mnist_train.py --num_hidden 1000,10 --weight_decay 0.001,0.1 --lr 1e-3 --decay 1.0 --save_weight_dir ../../data/approx-nn/saved_weights/mnist-1000-wz1-s5555-fix --save_log_dir logs/mnist-1000-wz1-s5555-fix --num_epoch 100 --train all --gpu 1 --first_train 1 --seed 5555

```