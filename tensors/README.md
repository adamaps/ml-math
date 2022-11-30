# Tensors

A machine learning-specific generalization of vectors and matrices to any number of dimensions.

Dimension|Mathematical Name|Description|Form
---|---|---|---
0|scalar|magnitude|$x$
1|vector|array|$[x_1, x_2, x_3]$
2|matrix|table|${\left\lbrack \matrix{x_{1,1} & x_{1,2} \cr x_{2,1} & x_{2,2}} \right\rbrack}$
3|3-tensor|3D-table/cube|
n|n-tensor|higher dimension|

## Scalars

Single numeric dimensionless value denoted by single italic lowercase letter e.g. $x$

Two common libraries are used for tensors, which include the ability to perform automatical differentiation. These are `PyTorch` and `tensorFlow`

### PyTorch

PyTorch tensors are designed to be pythonic, and to feel and behave like NumPy arrays. PyTorch tensors can be easily used for operations on GPUs (parallel matrix operations and training deep learning algorithms). Advantage of PyTorch is that is it much simpler to use and debug.

```python
import torch

# Produce a scalar tensor of 25 - optionally include dtype e.g. dtype=torch.float16
torch.tensor(25)

# Output
> tensor(25)
```

### TensorFlow

Created with a wrapper (e.g. `tf.Variable`, `tf.constant`, `tf.placeholder`, `tf.SparseTensor`). Easily convert to and from NumPy arrays and use. Advantage of TensorFlow is that it is more mature with greater support, and more flexibility for a wider variety of applications.

```python
import tensorflow

# Produce a scalar tensor of 25 - optionally include dtype e.g. dtype=tensorflow.float16
tensorflow.Variable(25)

# Output
> <tf.Variable 'Variable:0' shape=() dtype=int32, numpy=25>
```

## Vectors

One-dimensional array of numbers denoted by a single bold-italic lowercase letter e.g. $\boldsymbol{x} = [x_1, x_2, x_3]$

Common operation, transpose is as follows: $$[x_1, x_2, x_3]^T = {\left\lbrack \matrix{x_1 \cr x_2 \cr x_3} \right\rbrack}$$