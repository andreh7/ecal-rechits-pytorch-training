#!/usr/bin/env python

import torch.nn as nn
import torch

#----------------------------------------------------------------------

# implements Torch's JoinTable, see https://github.com/torch/nn/blob/master/doc/table.md#nn.JoinTable
class JoinTable(nn.Module):
    def __init__(self, dimension):
        super(JoinTable,self).__init__()

        self.dimension = dimension

    def forward(self,x):
        # expects x to be a list
        return torch.cat(x, self.dimension)

        
#----------------------------------------------------------------------
