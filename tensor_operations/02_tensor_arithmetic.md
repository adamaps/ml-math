

# Tensor Arithmetic

Adding and multiplying by scalar values to all elements in tensor.

```python
import torch

X_pt = torch.tensor([[25, 2], [5, 26], [3, 7]])

>  tensor([[25,  2],
        [ 5, 26],
        [ 3,  7]])

# Multiply by 2
X_pt*2

> tensor([[50,  4],
        [10, 52],
        [ 6, 14]])

# Add 2
X_pt+2

> tensor([[27,  4],
        [ 7, 28],
        [ 5,  9]])

# Can also use torch.mul() or torch.add()
torch.add(torch.mul(X_pt, 2), 2)

> tensor([[52,  6],
        [12, 54],
        [ 8, 16]])
```

When two tensors have the same dimension, the following multiplication method is **not matrix multiplication**, but rather the element-wise products also called the **Hadamard product**, denoted as:
$$A \odot X$$

```python
import torch

# Create another matrix as a scaled version of X
A = X_pt+2

> tensor([[27,  4],
        [ 7, 28],
        [ 5,  9]])

# Add the two same-dimension matrices
A + X_pt

> tensor([[52,  6],
        [12, 54],
        [ 8, 16]])

# Multiply the two same-dimension matrices
A * X_pt

> tensor([[675,   8],
        [ 35, 728],
        [ 15,  63]])