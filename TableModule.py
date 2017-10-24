#!/usr/bin/env python

import torch.nn as nn
import torch

#----------------------------------------------------------------------

# adapted from https://github.com/amdegroot/pytorch-containers#paralleltable
class TableModule(nn.Module):
    def __init__(self, layers):
        super(TableModule,self).__init__()

        for index, layer in enumerate(layers):
            self.add_module("module%d"% index, layer)

        self.layers = list(layers)

    def forward(self,x):
        # do not use self.modules() because 'self' might be included
        y = [ layer(xitem) for layer, xitem in zip(self.layers, x) ]
        
        return y
        
#----------------------------------------------------------------------
