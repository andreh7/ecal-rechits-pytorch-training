#!/usr/bin/env python

import torch.nn as nn

#----------------------------------------------------------------------

class PrintLayer(nn.Module):
    # a layer to just print the shape of its input arguments
    # the first n times (mostly for debugging / testing purposes)

    def __init__(self, nPrint = 1):
        super(PrintLayer, self).__init__()
        self.remaining = nPrint

    def forward(self, x):

        if self.remaining > 0:

            print x
            self.remaining -= 1

        return x

#----------------------------------------------------------------------
