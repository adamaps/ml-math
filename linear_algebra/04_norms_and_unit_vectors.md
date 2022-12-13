# Norms and Unit Vectors

Representing a `magnitude` and `direction` from origin. Norms are a class of function that allow us to quantify the magnitude (length) of a vector.

## L2 norm
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

## L1 norm
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

## Squared L2 norm - The dot product
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

## Max Norm
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

## Generalized Lp Norm
Formula to compute any norm type ($L^1$, $L^2$, etc), denoted as: $$||\boldsymbol{x}||_{p} = (\sum_i{|x_i|}^p)^{1/p}$$

where p must be:
- a real number
- greater than or equal to one

Norms, particularly $L^1$ and $L^2$ are used to regularize objective functions.