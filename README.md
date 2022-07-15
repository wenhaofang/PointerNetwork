## Pointer Network

This is Pointer Network for integer sorting problems.

Note: The project refers to [ast0414](https://github.com/ast0414/pointer-networks-pytorch)

Datasets:

* `dataset1`: Integer Sorting Dataset implemented by ourselves

Models:

* `model1`: [Pointer Networks](https://arxiv.org/abs/1506.03134)

### Unit Test

* for loader

```
PYTHONPATH=. python loaders/loader1.py
```

* for module

```
PYTHONPATH=. python modules/module1.py
```

### Main Process

```shell
python main.py
```

You can change the config either in the command line or in the file `utils/parser.py`

Here are the examples:

```shell
python main.py

python main.py --is_coverage
```
