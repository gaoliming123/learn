import torch
import torch.nn.functional as F

class GumbelSoftmax(torch.autograd.Function):
    """
    GumbelSoftmax:
        input: the probabilities and tensor shape is 1 x n
        output: one-hot when the limit of temperature is 0
    """
    @staticmethod
    def forward(ctx, distributions, temperature):
        uniform = torch.rand(* distributions.shape)
        g = -torch.log(-torch.log(uniform))
        y = (g + torch.log(distributions)) / temperature
        y = F.softmax(y, dim = 0)
        return  y

