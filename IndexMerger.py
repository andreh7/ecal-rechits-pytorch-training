#!/usr/bin/env python

import torch.nn as nn
import torch
from torch.autograd import Variable

#----------------------------------------------------------------------

# a module which can be used as first module to put together
# a minibatch of indices
#
# note that this can't be used inside a Sequential module because
# the Sequential module does not expect to be passed indices
# so we give the subsequent layers as an argument

class IndexMerger(nn.Module):

    def __init__(self, childModule):
        super(IndexMerger, self).__init__()

        self.childModule = childModule
        self.cudaDevice = None
        self.isCuda = False

    def cuda(self, device_id=None):
        self.cudaDevice = device_id
        self.isCuda = True

        self.childModule.cuda(device_id)

    def cpu(self):
        self.cudaDevice = None
        self.isCuda = False
        self.childModule.cpu()

    def forward(self, x, indices):
        # x is expected to be a list of numpy arrays
        # 
        # we convert the numpy arrays into torch tensors and those
        # into torch autograd Variables
        
        out = [ ] 

        for xitem in x:
            # numpy supports index selection from lists
            # so no need to use torch.cat
            
            tensor = torch.from_numpy(xitem[indices])
            
            if self.isCuda:
                tensor = tensor.cuda(self.cudaDevice)

            out.append(Variable(tensor))

        return self.childModule(out)

#----------------------------------------------------------------------
