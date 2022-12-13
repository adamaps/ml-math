# Tensor Reduction

Calculating the sum of all elements of a tensor is a common operation.

For a vector $\boldsymbol{x}$ of length $n$, we calculate $\sum^n_{i=1}x_i$

For a matrix $\boldsymbol{X}$ of dimension $m$ x $n$, we calculate $\sum^m_{i=1}\sum^n_{j=1}X_{i,j}$

```python
import torch

X = torch.tensor([[25, 2], [5, 26], [3, 7]])
> tensor([[25,  2],
       [ 5, 26],
       [ 3,  7]])

# Add all elements in X
torch.sum(X)
> tensor(68)

# Add all elements from specfic dimension in X
torch.sum(X, 0)
> tensor([33, 35])
```

Can also use `maximum`, `minimum`, `mean`, `product`