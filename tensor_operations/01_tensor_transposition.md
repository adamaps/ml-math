# Tensor Transposition

- Transpose of scalar is itself e.g. $x^T = x$
- Transpose of vector converts row to column and vice versa
- Transpose of matrix flips axes over main diagonal such that:

$$(\boldsymbol{X}^T)_{i,j} = \boldsymbol{X}_{j,i}$$

```python
import torch
x.T

import tensorflow
tensorflow.transpose(x)