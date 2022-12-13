# The dot product

The dot product is a common operation in machine learning and deep learning in particular. Two vectors must have same length $n$. We can calculate the dot product between them. The following all represent the same thing:

$x \cdot y$

$x^Ty$ 

$\langle{x},y\rangle$

Calculate the product of two matrices in element-wise fashion. 1st element in $x$ multiplied by 1st element in $y$, plus the 2nd element in $x$ multiplied by 2nd element in $y$, etc. I.e.:
$$x \cdot y = \sum^n_{i=1}x_iy_i$$

This results in a single scalar value.

```python
import torch

x = torch.tensor([25., 2., 5.])
y = torch.tensor([0., 1., 2.])

# note, must be float values, not integers
torch.dot(x,y)
> 12
```