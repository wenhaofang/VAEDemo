## VAE Demo

This repository includes some demo VAE models.

Datasets:

* `dataset1`: [MNIST](http://yann.lecun.com/exdb/mnist/)

Models:

* `model3`: VAE

* `model4`: CVAE

### Unit Test

* for loaders

```shell
PYTHONPATH=. python loaders/loader1.py
```

* for modules

```shell
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
# module3
python main.py --module 3
```

```shell
# module4
python main.py --module 4
```
