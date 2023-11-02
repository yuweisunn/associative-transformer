# Associative Transformer Is A Sparse Representation Learner
# See https://arxiv.org/abs/2309.12862
#
# Author: Yuwei Sun


import torch, random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# Define your custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]


class IdentityLayer(nn.Module):
                def __init__(self):
                    super(IdentityLayer, self).__init__()

                def forward(self, x):
                    return x


def get_energy(R, Y, beta):
    lse = -(1.0/beta)*torch.logsumexp(beta*(torch.bmm(R, Y.transpose(1,2))), dim=2) # -lse(beta, Y^T*R)
    lnN = (1.0/beta)*torch.log(torch.tensor(Y.shape[1], dtype=float)) # beta^-1*ln(N)
    RTR = 0.5*torch.bmm(R, R.transpose(1,2)) # R^T*R
    M = 0.5*((torch.max(torch.linalg.norm(Y, dim=2), dim=1))[0]**2.0) # 0.5*M^2  *very large value*
    energy = lse + lnN + RTR + M
    return energy