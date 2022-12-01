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

Two common libraries are used for tensors, which include the ability to perform automatical differentiation. These are `PyTorch` and `TensorFlow`

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

One-dimensional array of numbers denoted by a single bold-italic lowercase letter e.g. $\boldsymbol{x} = [x_1, x_2, x_3]$.
Vectors represent a point in space.
- A vector of length 2 represents a point in 2D space.
- A vector of length 3 represents a point in 3D space.
- A vector of length n represents a point in n-dimensional space.

E.g. a vector $\boldsymbol{x} = [x_1, x_2] = [12, 4]$ can be represented as a line vector drawn from point $(0,0)$ to point $(12,4)$.

Common operation, transpose $n \cdot m$ to $m \cdot n$ as follows: $$[x_1, x_2, x_3]^T = {\left\lbrack \matrix{x_1 \cr x_2 \cr x_3} \right\rbrack}$$

### Numpy

```python
import numpy

# Produce a vector of size 3 (just a length) - optionally include dtype e.g. dtype=numpy.float16
x = np.array([25, 2, 5])

# You can only transpose a 1-dimensional array if you nest the array in the first element
# of a multi-dimensional array
y = np.array([[25, 2, 5]]).T

# Output
> array([[25],
         [ 2],
         [ 5]])
```

### PyTorch
```python
import torch

# Produce a vector tensor - optionally include dtype e.g. dtype=torch.float16
torch.tensor([25, 2, 5])

# Output
> tensor([25,  2,  5])
```

### TensorFlow
```python
import tensorflow

# Produce a vector tensor - optionally include dtype e.g. dtype=torch.float16
tensorflow.Variable([25, 2, 5])

# Output
> <tf.Variable 'Variable:0' shape=(3,) dtype=int32, numpy=array([25,  2,  5], dtype=int32)>
```

## Norms and Unit Vectors

Representing a `magnitude` and `direction` from origin. Norms are a class of function that allow us to quantify the magnitude (length) of a vector.

### L2 norm
The most common norm vector is the $L^2$ norm, which is the square root of the sum of squares in a vector, denoted as: $$||\boldsymbol{x}||_{2} = \sqrt{\sum_ix_i^2}$$

The norm vector measures the simple (Euclidean) distance from the origin (zero vector), and is the most common norm in machine learning. Also denoted as $||\boldsymbol{x}||$ instead of $||\boldsymbol{x}||_{2}$.

### Python
```python
import numpy

x = np.array([25, 2, 5])

# Use numpy linear algebra norm function to produce the L2 norm of a vector
numpy.linalg.norm(x)

# Output - L2 norm is the magnitude of the vector x - with units matching the units of x
> 25.573423705088842
```

### Unit vectors
Special case of vector where the vector length ($L^2$ norm) is equal to 1.

### L1 norm
$L^1$ norm is the sum of the absolute value in the vector $\boldsymbol{x}$. The value varies linearly at all locations, near or far from origin, and is used whenever the differences between zero and non-zero values are key, denoted as: $$||\boldsymbol{x}||_{1} = \sum_i{|x_i|}$$

### Python
```python
import numpy

x = np.array([25, 2, 5])

# Use numpy linear algebra norm function to produce the L2 norm of a vector
numpy.abs(x[0]) + numpy.abs(x[1]) + numpy.abs(x[2])

# Output - L2 norm is the magnitude of the vector x - with units matching the units of x
> 32
```

### Squared L2 norm - The dot product
Same as the $L^2$ norm, but without taking the final square root, denoted as $$||\boldsymbol{x}||_{2}^2 = \sum_i|x_i|^2$$

Computationally cheaper than $L^2$ norm because:
- The squared $L^2$ norm is simply $\boldsymbol{x}^T\boldsymbol{x}$
- The derivative (used to train ML algorithms) of element $x$ within $\boldsymbol{x}$ requires that element alone, rather than information about all the elements in the vector.
- The downside is that the value grows slowly near the origin, so is not good at differentiating between zero and non-zero values.

### Python
```python
import numpy

x = np.array([25, 2, 5])

# Use numpy dot product to compute x^Tx (Squared L2 norm) of a vector
numpy.dot(x)

# Output - L2 norm is the magnitude of the vector x - with units matching the units of x
> 654
```

### Max Norm
Return the maximum value of each of the elements in $\boldsymbol{x}$, denoted as: $$||\boldsymbol{x}||_{\infty} = max_i|x_i|$$

### Python
```python
import numpy

x = numpy.array([25, 2, 5])

# Use numpy dot product to compute x^Tx (Squared L2 norm) of a vector
numpy.max([numpy.abs(25), numpy.abs(2), numpy.abs(5)])

# Output - L2 norm is the magnitude of the vector x - with units matching the units of x
> 25
```

### Generalized Lp Norm
Formula to compute any norm type ($L^1$, $L^2$, etc), denoted as: $$||\boldsymbol{x}||_{p} = (\sum_i{|x_i|}^p)^{1/p}$$

where p must be:
- a real number
- greater than or equal to one

Norms, particularly $L^1$ and $L^2$ are used to regularize objective functions.