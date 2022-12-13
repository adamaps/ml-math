# Generic and Higher Rank Tensors
- Denoted with uppercase, bold, italics and sans-serif e.g. $\boldsymbol{X}_{i,j,k,l}$
- Rank 4 tensors are common for images, where each dimension corresponds to:
1. Number of images in a training batch e.g. 32
2. Image height in pixels e.g. 28
3. Image width in pixels e.g. 28
4. Number of colour channgels e.g. 3 for full-colour RGB

```python
import torch

# Create empty tensor of size (32, 28, 28, 3)
images = torch.zeros([32, 28, 28, 3])

# Output
> tensor([[[[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.],
          ...,
          [0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]],

         [[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.],
          ...,
          [0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]],

         [[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.],
          ...,
          [0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]],

         ...,

         etc.
```