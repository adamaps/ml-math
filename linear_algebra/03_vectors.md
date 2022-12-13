# Vectors

One-dimensional array of numbers denoted by a single bold-italic lowercase letter e.g. $\boldsymbol{x} = [x_1, x_2, x_3]$.
Vectors represent a point in space.
- A vector of length 2 represents a point in 2D space.
- A vector of length 3 represents a point in 3D space.
- A vector of length n represents a point in n-dimensional space.

E.g. a vector $\boldsymbol{x} = [x_1, x_2] = [12, 4]$ can be represented as a line vector drawn from point $(0,0)$ to point $(12,4)$.

Common operation, transpose $n \cdot m$ to $m \cdot n$ as follows: $$[x_1, x_2, x_3]^T = {\left\lbrack \matrix{x_1 \cr x_2 \cr x_3} \right\rbrack}$$

## Numpy

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

## PyTorch
```python
import torch

# Produce a vector tensor - optionally include dtype e.g. dtype=torch.float16
torch.tensor([25, 2, 5])

# Output
> tensor([25,  2,  5])
```

## TensorFlow
```python
import tensorflow

# Produce a vector tensor - optionally include dtype e.g. dtype=torch.float16
tensorflow.Variable([25, 2, 5])

# Output
> <tf.Variable 'Variable:0' shape=(3,) dtype=int32, numpy=array([25,  2,  5], dtype=int32)>
```