#!/usr/bin/env python

import torch.nn as nn

#----------------------------------------------------------------------

# see also https://gist.github.com/lebedov/062d0cd8e51d00e2dba209cff9c492fc
# and https://discuss.pytorch.org/t/non-legacy-view-module/131
# and https://github.com/pytorch/tutorials/blob/345354a5050a59172121ed3201612e56da8d1bfc/beginner_source/former_torchies/nn_tutorial.py#L89

import torch.autograd
class View(nn.Module):

    def __init__(self, *sizes):
        super(View, self).__init__()
        self.sizes = tuple(sizes)
        
    def forward(self,x):
        return x.view(x.size(0), # keep minibatch dimension
                      *self.sizes)

    def __repr__(self):
        return self.__class__.__name__ + ' (sizes=%s)' % str(self.sizes)

#----------------------------------------------------------------------
