#!/usr/bin/env python

import torch.nn as nn
import torch

#----------------------------------------------------------------------

# adapted from https://github.com/amdegroot/pytorch-containers#paralleltable
# see https://github.com/torch/nn/blob/master/doc/table.md#nn.ParallelTable
# what the Torch module does
class ParallelTable(nn.Module):
    def __init__(self, layers):
        super(ParallelTable, self).__init__()

        for index, layer in enumerate(layers):
            self.add_module("module%d"% index, layer)

        self.layers = list(layers)

    def forward(self,x):
        # do not use self.modules() because 'self' might be included
        y = [ layer(xitem) for layer, xitem in zip(self.layers, x) ]
        
        return y
        
#----------------------------------------------------------------------
