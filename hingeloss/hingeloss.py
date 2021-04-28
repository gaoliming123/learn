import torch.nn as nn
import torch

class SmoothedHingeLoss(nn.Module):
    def __init__(self, device):
        self.device = device
        super(SmoothedHingeLoss, self).__init__()

   def forward(self, pos, neg):
       # x >= 1 h(x) == 0; x < 0 h(x) = 1/2 - x; (1 - x)^2 / 2
       x  = pos - neg
       x_ = x.clone().detach()
       one = torch.tensor(1.).to(self.device)
       zero = torch.tensor(0.).to(self.device)
       point = torch.tensor(.5).to(self.device)
       two = torch.tensor(2.).to(self.device)
       x1 = x[x_ >= one]
      x1 = torch.zeros(*x1.shape).to(self.device)
       x2 = x[x_ < zero]
       x2 = point - x2
       x3 = x[x_ >= zero]
       x3 = x3[x3 < one]
       x3 = ((one - x3) ** two) / two
       loss = (x1.sum() + x2.sum() + x3.sum()) / (len(x1) + len(x2) + len(x3))
       return loss

class OrginHingeLoss(nn.Module):
     def __init__(self, device):
         self.device = device
         super(OrginHingeLoss, self).__init__()
 
     def forward(self, pos, neg):
         y = torch.cat([torch.ones(*pos.shape).to(self.device), - torch.ones(*neg.shape).to(self.device)], dim=0)
         score = torch.cat([pos, neg], dim=0)
         loss = torch.clamp(torch.tensor(1.).to(self.device) - y * score, min=0).mean()
         return loss

class HingeMarginLoss(nn.Module):
     def __init__(self, device):
         self.device = device
         super(HingeMarginLoss, self).__init__()
 
     def forward(self, pos, neg):
         y = torch.cat([torch.ones(*pos.shape).to(self.device), - torch.ones(*neg.shape).to(self.device)], dim=0)
         score = torch.cat([pos, neg], dim=0)
         loss = torch.clamp(torch.tensor(1.).to(self.device) - y * score, min=0).mean()
         return loss
