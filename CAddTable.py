#!/usr/bin/env python

import torch.nn as nn

#----------------------------------------------------------------------

# see https://github.com/torch/nn/blob/master/doc/table.md#nn.CAddTable
# and https://github.com/pytorch/tutorials/blob/master/beginner_source/former_torchies/nn_tutorial.py
class CAddTable(nn.Module):

    def __init__(self):
        super(CAddTable,self).__init__()

    def forward(self,x):
        # x should be a list
        return sum(x)

    def __repr__(self):
        return self.__class__.__name__ + ' ()'

#----------------------------------------------------------------------

