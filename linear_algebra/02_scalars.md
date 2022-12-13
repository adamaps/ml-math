# Scalars

Single numeric dimensionless value denoted by single italic lowercase letter e.g. $x$

Two common libraries are used for tensors, which include the ability to perform automatical differentiation. These are `PyTorch` and `TensorFlow`

## PyTorch

PyTorch tensors are designed to be pythonic, and to feel and behave like NumPy arrays. PyTorch tensors can be easily used for operations on GPUs (parallel matrix operations and training deep learning algorithms). Advantage of PyTorch is that is it much simpler to use and debug.

```python
import torch

# Produce a scalar tensor of 25 - optionally include dtype e.g. dtype=torch.float16
torch.tensor(25)

# Output
> tensor(25)
```

## TensorFlow

Created with a wrapper (e.g. `tf.Variable`, `tf.constant`, `tf.placeholder`, `tf.SparseTensor`). Easily convert to and from NumPy arrays and use. Advantage of TensorFlow is that it is more mature with greater support, and more flexibility for a wider variety of applications.

```python
import tensorflow

# Produce a scalar tensor of 25 - optionally include dtype e.g. dtype=tensorflow.float16
tensorflow.Variable(25)

# Output
> <tf.Variable 'Variable:0' shape=() dtype=int32, numpy=25>
```