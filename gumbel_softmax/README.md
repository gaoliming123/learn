# pytorch-Gumbel\_Softmax
Gumbel Softmax采样算法

## 讲解

[漫谈重参数：从正态分布到Gumbel Softmax](https://www.spaces.ac.cn/archives/6705#%E5%9F%BA%E6%9C%AC%E6%A6%82%E5%BF%B5)

## Usage

使用方法，引入``GumbelSoftmax``类即可
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
