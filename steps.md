## Steps w.r.t two-level Neural Networks

```
python mnist_train.py --num_hidden 10000,10 --weight_decay 0.0,0.0 --lr 1e-3 --save_weight_dir ../../data/approx-nn/saved_weights/mnist-10000 --save_log_dir logs/mnist-10000
```

- default training:
    - m=1000, seed=1234: 100/98
    - 
```
python mnist_train.py --num_hidden 1000,10 --weight_decay 0.001,0.1 --lr 1e-3 --decay 1.0 --save_weight_dir ../../data/approx-nn/saved_weights/mnist-1000 --save_log_dir logs/mnist-1000 --num_epoch 100 --train all
python mnist_train.py --num_hidden 100,10 --weight_decay 0.001,0.1 --lr 1e-3 --decay 1.0 --save_weight_dir ../../data/approx-nn/saved_weights/mnist-100 --save_log_dir logs/mnist-100 --num_epoch 100 --train all
```

- could two-level nn with random first layer learn competitive results compared with original neural networks
    - m=10000: 98/96
    - m=1000: 97/95

```
python mnist_train.py --num_hidden 1000,10 --weight_decay 0.0,0.0 --lr 1e-3 --decay 1.0 --save_weight_dir ../../data/approx-nn/saved_weights/mnist-1000-lst --save_log_dir logs/mnist-1000-lst --num_epoch 500 --train lst
```

- could the distribution of two overaparmaterized neural network be similar?
    - 