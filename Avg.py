#!/usr/bin/env python

import torch.nn as nn

#----------------------------------------------------------------------
class Avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(Avg, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        return nn.functional.avg_pool2d(x, self.kernel_size, self.stride)

    def __repr__(self):
        return self.__class__.__name__ + ' (kernel_size=%s, stride=%s)' % (str(self.kernel_size), str(self.stride))

#----------------------------------------------------------------------
