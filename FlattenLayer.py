#!/usr/bin/env python
import torch.nn as nn

class FlattenLayer(nn.Module):
    # a 'View' reshaping the input to dimension (minibatch, product of remaining dimensions)
    # typically to be used after a convolutional network and before the dense layers

    ### def __init__(self):
    ###     super(FlattenLayer, self).__init__()

    def forward(self, x):
        # see e.g. https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py#L44
        return x.view(x.size(0), -1)

    def __repr__(self):
        return self.__class__.__name__ + " ()"

#----------------------------------------------------------------------
