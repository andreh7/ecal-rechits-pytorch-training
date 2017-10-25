#!/usr/bin/env python

import torch.nn as nn

#----------------------------------------------------------------------

class Marker(nn.Module):
    # a module to show a marker string in a list of modules
    # (mostly for debugging purposes)
    
    def __init__(self, markerText):
        super(Marker, self).__init__()

        self.markerText = markerText

    def forward(self, x):
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (%s)' % self.markerText

#----------------------------------------------------------------------
