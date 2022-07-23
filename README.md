## VAE Demo

This repository includes some demo VAE models.

Note: The project refers to [YixinChen-AI](https://github.com/YixinChen-AI/CVAE-GAN-zoos-PyTorch-Beginner) and [timbmg](https://github.com/timbmg/VAE-CVAE-MNIST)

Datasets:

* `dataset1`: [MNIST](http://yann.lecun.com/exdb/mnist/)

Models:

* `model1`: AE

* `model2`: DAE

* `model3`: VAE

* `model4`: CVAE

### Unit Test

* for loaders

```shell
PYTHONPATH=. python loaders/loader1.py
```

* for modules

```shell
# AE
PYTHONPATH=. python modules/module1.py
# DAE
PYTHONPATH=. python modules/module2.py
# VAE
PYTHONPATH=. python modules/module3.py
# CVAE
PYTHONPATH=. python modules/module4.py
```

### Main Process

```shell
python main.py
```

You can change the config either in the command line or in the file `utils/parser.py`

Here are the examples for each module:

```shell
# module1
python main.py \
    --name 1 \
    --module 1
```

```shell
# module2
python main.py \
    --name 2 \
    --module 2
```

```shell
# module3
python main.py \
    --name 3 \
    --module 3
```

```shell
# module4
python main.py \
    --name 4 \
    --module 4
```

### Note (重点、难点、疑点、TODO、...)

1、目前 Encoder 和 Decoder 都是用 MLP 架构，对于 AE 和 DAE，当 latent_size 等于 2 时，降维效果并不理想，如果想要提升，可以使用卷积替代
