# Basis, Orthogonal, and Orthonormal Vectors

## Basis
- Can be scale to represent __any__ vector in a given vector space.
- Typically use unit vectors along axes of cector space. If $i$ and $j$ represent the unit vectors along the $x$ and $y$ axes, respectively, then those same unit vectors can be scaled by any amount along those axes to create a new vector. E.g.
$$v = 1.5i + 2j$$

## Orthogonal
- Where the dot product of $\boldsymbol{x}$ and $\boldsymbol{y}$ is zero. I.e. $\boldsymbol{x}^T\boldsymbol{y} = 0$. In other words, vectors $\boldsymbol{x}$ and $\boldsymbol{y}$ are at 90° to each other
- n-dimensional space has max $n$ muturally orthogonal vectors. E.g. 2D space has at most 2 orthogonal vectors.
- Othornormal vectors are othogonal __and__ all have unit norm. E.g. the $i$ and $j$ unit vectors of length 1 are both unit vectors and orthogonal vectors, and are therefore orthonormal vectors.

```python
import numpy

i = np.array([1, 0])
j = np.array([0, 1])
numpy.dot(i, j)

# Output
> 0
```