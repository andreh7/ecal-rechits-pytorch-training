#!/usr/bin/env python

import torch.nn as nn

#----------------------------------------------------------------------

# see https://github.com/torch/nn/blob/master/doc/containers.md#nn.Concat
# and https://github.com/torch/nn/blob/master/Concat.lua
class Concat(nn.Module):

    def __init__(self, dimension, modules):
        super(Concat,self).__init__()

        for index, module in enumerate(modules):
            self.add_module("module%d"% index, module)

        self.dimension = dimension
        self.mods = list(mods)

    def forward(self,x):
        # apply each module to x
        # and concatenate their output along the given dimension
        y = [ module(x) for modules in self.mods ]
        
        return torch.cat(self.dimension, y)


#----------------------------------------------------------------------
