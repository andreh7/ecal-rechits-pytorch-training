#!/usr/bin/env python

# Reshape/View layer for pytorch, from https://discuss.pytorch.org/t/what-is-reshape-layer-in-pytorch/1110/8

import torch.nn as nn

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
