import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np

class DKGumbel(nn.Module):
    def __init__(self, nentity, nrelation, emb_dim, device, temperature):
        super(DKGumbel, self).__init__()
        self.device    = device
        self.emb_dim   = emb_dim
        self.nentity   = nentity
        self.nrelation = nrelation
        self.k         = 2
        self.dihedrons = 8
        self.temperature = temperature

        self.entity_embedding   = nn.Parameter(torch.zeros(self.nentity, self.emb_dim, 1, self.k))
        nn.init.uniform_(tensor = self.entity_embedding, a = - 1, b = 1)
        self.dihedron           = nn.Parameter(torch.zeros(self.dihedrons, self.k * 2), requires_grad = False)
        self.init_dihedron()
        self.relation_embedding = nn.Parameter(torch.zeros(self.nrelation, self.emb_dim, self.dihedrons))
        nn.init.uniform_(tensor = self.relation_embedding, a = - 1, b = 1)

    def l2_loss(self):
        l2_loss = (self.entity_embedding ** 2).mean() + (self.relation_embedding ** 2).mean()
        return l2_loss / 2

    def forward(self, head_idx, relation_idx, tail_idx):
        batch_size = head_idx.shape[0]
        real_shape = batch_size * self.emb_dim
        heads      = torch.index_select(self.entity_embedding, dim = 0, index = head_idx)
        tails      = torch.index_select(self.entity_embedding, dim = 0, index = tail_idx)

        relations  = torch.index_select(self.relation_embedding, dim = 0, index = relation_idx)
        relations  = F.softmax(relations, dim = 2)
        relations  = F.gumbel_softmax(relaitons, tau = self.temperature, hard = True)
        vectors    = relations.view(batch_size, self.emb_dim, 8, 1) * self.dihedron.unsqueeze(dim = 0).unsqueeze(dim = 0).repeat(batch_size, self.emb_dim, 1, 1)
        vectors.permute(0, 1, 3, 2).sum(dim = 3)
        relation_matrices = torch.zeros(batch_size, self.emb_dim, self.k, self.k).to(self.device)
        relation_matrices[:, :, 0, 0] = vectors[:, :, 0]
        relation_matrices[:, :, 0, 1] = vectors[:, :, 1]
        relation_matrices[:, :, 1, 0] = vectors[:, :, 2]
        relation_matrices[:, :, 1, 1] = vectors[:, :, 3]

        score = torch.bmm(
                    torch.bmm(heads.view(real_shape, 1, self.k), relation_matrices.view(real_shape, self.k, self.k)),
                    tails.view(real_shape, 1, self.k).permute(0, 2, 1)
                    ).view(batch_size, self.emb_dim)
        score = torch.norm(score, p = 2, dim = 1)

        return score

    def init_dihedron(self):
        self.dihedron.data[0][0] = torch.tensor(1.)
        self.dihedron.data[0][3] = torch.tensor(1.)
        self.dihedron.data[1][1] = torch.tensor(-1.)
        self.dihedron.data[1][2] = torch.tensor(1.)
        self.dihedron.data[2][0] = torch.tensor(-1.)
        self.dihedron.data[2][3] = torch.tensor(-1.)
        self.dihedron.data[3][1] = torch.tensor(1.)
        self.dihedron.data[3][2] = torch.tensor(-1.)
        self.dihedron.data[4][0] = torch.tensor(1.)
        self.dihedron.data[4][3] = torch.tensor(-1.)
        self.dihedron.data[5][1] = torch.tensor(1.)
        self.dihedron.data[5][2] = torch.tensor(1.)
        self.dihedron.data[6][0] = torch.tensor(-1.)
        self.dihedron.data[6][3] = torch.tensor(1.)
        self.dihedron.data[7][1] = torch.tensor(-1.)
        self.dihedron.data[7][3] = torch.tensor(-1.)
        '''
        self.dihedron.data[0][0][0] = torch.tensor(1.)
        self.dihedron.data[0][1][1] = torch.tensor(1.)
        self.dihedron.data[1][0][1] = torch.tensor(-1.)
        self.dihedron.data[1][1][0] = torch.tensor(1.)
        self.dihedron.data[2][0][0] = torch.tensor(-1.)
        self.dihedron.data[2][1][1] = torch.tensor(-1.)
        self.dihedron.data[3][0][1] = torch.tensor(1.)
        self.dihedron.data[3][1][0] = torch.tensor(-1.)
        self.dihedron.data[4][0][0] = torch.tensor(1.)
        self.dihedron.data[4][1][1] = torch.tensor(-1.)
        self.dihedron.data[5][0][1] = torch.tensor(1.)
        self.dihedron.data[5][1][0] = torch.tensor(1.)
        self.dihedron.data[6][0][0] = torch.tensor(-1.)
        self.dihedron.data[6][1][1] = torch.tensor(1.)
        self.dihedron.data[7][0][1] = torch.tensor(-1.)
        self.dihedron.data[7][1][0] = torch.tensor(-1.)
        '''

