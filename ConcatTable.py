#!/usr/bin/env python

import torch.nn as nn
import torch

#----------------------------------------------------------------------

# see https://github.com/torch/nn/blob/master/doc/table.md#nn.ConcatTable
# what it did in lua Torch
#
# "ConcatTable is a container module that applies each member module to the same input Tensor or table."

class ConcatTable(nn.Module):
    
    def __init__(self, modules):
        super(ConcatTable,self).__init__()

        for index, module in enumerate(modules):
            self.add_module("module%d"% index, module)

        self.mods = list(modules)

    def forward(self,x):
        # apply each module to x
        # do not use self.modules() because 'self' might be included
        y = [ module(x) for module in self.mods ]
        
        return y
        
#----------------------------------------------------------------------
