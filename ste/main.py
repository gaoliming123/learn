import torch
from LBSign import LBSign

if __name__ == '__main__':

    sign = LBSign.apply
    params = torch.randn(4, requires_grad = True)
    output = sign(params)
    loss = output.mean()
    loss.backward()
    print (params.grad)
