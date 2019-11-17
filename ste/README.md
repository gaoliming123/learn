# pytorch-LBSign

## 讲解
[pytorch实现简单的straight-through estimator(STE)](https://gaoliming123.github.io/2019/11/13/ste/)

## Usage
引入``LBSign``类即可
```
from LBSign import LBSign

sign = LBSign.apply
params = torch.randn(4, requires_grad = True)
output = sign(params)
loss = output.mean()
loss.backward()
```
