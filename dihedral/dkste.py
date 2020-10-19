import torch
import torch.nn as nn
import pdb
import numpy as np
from LBSign import LBSign

class DKSTE(nn.Module):
    def __init__(self, nentity, nrelation, emb_dim, device):
        super(DKSTE, self).__init__()
        self.device    = device
        self.sign      = LBSign.apply
        self.emb_dim   = emb_dim
        self.nentity   = nentity
        self.nrelation = nrelation
        self.k         = 2

        self.entity_embedding = nn.Parameter(torch.zeros(self.nentity, self.emb_dim, 1, self.k))
        nn.init.uniform_(tensor = self.entity_embedding, a = - 1, b = 1)
        self.relation_embedding = nn.Parameter(torch.zeros(self.nrelation, self.emb_dim, self.k))
        nn.init.uniform_(tensor = self.relation_embedding, a = - 1, b = 1)
        self.alpha_embedding = nn.Parameter(torch.zeros(self.nrelation, self.emb_dim))
        nn.init.uniform_(tensor = self.alpha_embedding, a = - 1, b = 1)

    def l2_loss(self):
        l2_loss = (self.entity_embedding ** 2).mean() + (self.relation_embedding ** 2).mean() + (self.alpha_embedding ** 2).mean()
        return l2_loss / 3

    def forward(self, head_idx, relation_idx, tail_idx):
        batch_size = head_idx.shape[0]
        real_shape = batch_size * self.emb_dim
        heads      =  torch.index_select(self.entity_embedding, dim = 0, index = head_idx)
        tails      =  torch.index_select(self.entity_embedding, dim = 0, index = tail_idx)
        relations  =  torch.index_select(self.relation_embedding, dim = 0, index = relation_idx)
        alphas     =  torch.index_select(self.alpha_embedding, dim = 0, index = relation_idx)

        x, y, alphas      = self.sign(relations[:, :, 0]), self.sign(relations[:, :, 1]), self.sign(alphas)
        relation_matrices = torch.zeros(batch_size, self.emb_dim, self.k, self.k).to(self.device)
        relation_matrices[:, :, 0, 0] = (x + y) / 2
        relation_matrices[:, :, 0, 1] = (x - y) / 2 * -alphas
        relation_matrices[:, :, 1, 0] = (x - y) / 2
        relation_matrices[:, :, 1, 1] = (x + y) / 2 * alphas

        score = torch.bmm(
                    torch.bmm(heads.view(real_shape, 1, self.k), relation_matrices.view(real_shape, self.k, self.k)),
                    tails.view(real_shape, 1, self.k).permute(0, 2, 1)
                    ).view(batch_size, self.emb_dim)
        score = torch.norm(score, p = 2, dim = 1)

        return score
