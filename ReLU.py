#!/usr/bin/env python

import torch.nn as nn
import torch.nn.functional as F

#----------------------------------------------------------------------

class ReLU(nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        # TODO: could we use inplace ?
        return F.relu(x)

    def __repr__(self):
        return self.__class__.__name__ + ' ()'

#----------------------------------------------------------------------
