# pytorch-LBSign

##
使用``torch.sign``的时候如果我们希望参数的梯度不是0而是可以对参数进行梯度更新，如果更新算法是根据``straight-through estimator``中如下公式更新:

![](./images/0.png)
## Usage

demo

```
from LBSign import LBSign

sign = LBSign.apply
params = torch.randn(4, requires_grad = True)
output = sign(params)
loss = output.mean()
loss.backward()
```
