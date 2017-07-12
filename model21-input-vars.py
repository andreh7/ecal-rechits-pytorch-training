#!/usr/bin/env python

# model just returning an input variable for calculation of the AUC

#----------------------------------------------------------------------
# (default) model parameters
#----------------------------------------------------------------------

# size and number of hidden layers on input side

#----------------------------------------------------------------------
# function to prepare input data samples
#----------------------------------------------------------------------

selectedVariables = [ 'phoIdInput/pfChgIso03worst' ]

def makeInput(dataset, rowIndices, inputDataIsSparse):
    assert inputDataIsSparse,"non-sparse input data is not supported"

    return - dataset['input'][rowIndices][:,0]

#----------------------------------------------------------------------

import torch.nn as nn
import torch
from torch.autograd import Variable

class Net(nn.Module):
    
    def __init__(self):
        
        super(Net, self).__init__()

    #----------------------------------------
    
    def forward(self, dataset, indices):

        
        return Variable(torch.FloatTensor(dataset[indices]))


    #----------------------------------------

    def getNumOutputNodes(self):
        # do not apply any loss function
        return None

#----------------------------------------------------------------------

def makeModel():

    return Net()

#----------------------------------------------------------------------

