# pytorch-Gumbel\_Softmax


## Usage

demo
```
>>> from gumbel_softmax import GumbelSoftmax
>>> import torch
>>> 
>>> input = torch.tensor([0.1, 0.2, 0.5, 0.2])
>>> gumbel_softmax = GumbelSoftmax.apply
>>> y = gumbel_softmax(input, 0.0001)
>>> y
>>> tensor([0., 0., 1., 0.])
```
