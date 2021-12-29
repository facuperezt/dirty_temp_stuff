import numpy as np
import torch.nn
from torch.nn.functional import relu

import dataLoader
def cn(src,tar,adj):
    row_src = adj[src]
    row_tar = adj[tar]

    n = 0
    for i in range(row_tar.shape[0]):
        if row_src[i] == row_tar[i] :
            n+= 1

def create_Dataset(self, adj, data):
    for i in range(adj.shape[0]):


class MLP_base(torch.nn.Module):

    def __init__(self):
        super(MLP_base, self).__init__()  # from torch documentation TODO look up what it does
        self.input = torch.nn.Linear(6, 6)
        self.hidden = torch.nn.Linear(6, 6)
        self.output = torch.nn.Linear(6, 1)

    def forward(self,x):
        h = self.input(x)

        X = relu(h)
        h = self.hidden(X)

        X = relu(h)
        h = self.output(X)

        return torch.sigmoid(h)
