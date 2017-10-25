#!/usr/bin/env python

import torch.nn as nn

#----------------------------------------------------------------------

# we can't use torch.legacy.nn.Identity() because it does not inherit
# from nn.Module (but from torch.legacy.nn.Module)

class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

    def __repr__(self):
        return self.__class__.__name__ + " ()"

#----------------------------------------------------------------------

