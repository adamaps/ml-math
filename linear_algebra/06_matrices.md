# Matrix Tensors
- Two dimensional array of numbers
- Denoted in uppercase, italics and bold. e.g. $\boldsymbol{X}$
- Notation based on height then width e.g. $(n_{row}, n_{col})$
- A matrix with three rows and 2 columns is denoted (3, 2)
- Individual scalar elements denoted in uppercase italics only e.g.
$${\left\lbrack \matrix{X_{1,1} & X_{1,2} \cr X_{2,1} & X_{2,2}} \right\rbrack}$$

- Full rows or columns can be represented using colons e.g. all elements of column 1 is $X_{:,1}$ and all elements of row 2 is $X_{2,:}$.


## Python
```python
import numpy

# Note the nested brackets
X = np.array([[25, 2], [5, 26], [3, 7]])

# Get shape
X.shape
> (3,2)

# Size = columns x rows
X.size
> 6

# All elements of first column
X[:,0]
> array([25,  5,  3])

# All elements of middle row
X[1,:]
> array([ 5, 26])

# Slicing by-index
X[0:2, 0:2]
> array([[25,  2],
       [ 5, 26]])
```

## PyTorch
```python
import torch
X_pt = torch.tensor([[25, 2], [5, 26], [3, 7]])
> tensor([[25,  2],
        [ 5, 26],
        [ 3,  7]])

X_pt.shape # more pythonic
> torch.Size([3, 2])

X_pt[1,:]
> tensor([ 5, 26])
```

## Tensorflow
```python
import tensorflow
X_tf = tensorflow.Variable([[25, 2], [5, 26], [3, 7]])
> <tf.Variable 'Variable:0' shape=(3, 2) dtype=int32, numpy=
array([[25,  2],
       [ 5, 26],
       [ 3,  7]], dtype=int32)>
tf.rank(X_tf)
> <tf.Tensor: shape=(), dtype=int32, numpy=2>

tf.shape(X_tf)
> <tf.Tensor: shape=(2,), dtype=int32, numpy=array([3, 2], dtype=int32)>

X_tf[1,:]
> <tf.Tensor: shape=(2,), dtype=int32, numpy=array([ 5, 26], dtype=int32)>
```