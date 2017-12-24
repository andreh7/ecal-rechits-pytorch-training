#!/usr/bin/env python

# like model06 but with dropout layer only applied
# to the rechits variables, not the other (track iso)
# variables


import rechitmodelutils

import numpy as np
import math

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# see http://torch.ch/blog/2016/02/04/resnets.html
# and https://discuss.pytorch.org/t/pytorch-performance/3079

# this seems not to exist in PyTorch ?
# torch.backends.cudnn.fastest = True

torch.backends.cudnn.benchmark = True

rechits_dim = (7,23)

#----------------------------------------------------------------------
# model
#----------------------------------------------------------------------

# see https://discuss.pytorch.org/t/pixelwise-weights-for-mseloss/1254
# and http://pytorch.org/docs/master/_modules/torch/nn/modules/loss.html#MSELoss
# and http://pytorch.org/docs/master/_modules/torch/nn/modules/loss.html#BCELoss
class WeightedMSELoss(torch.nn.modules.loss._WeightedLoss):

    def forward(self, input, target):
        torch.nn.modules.loss._assert_no_grad(target)

        # input[0] are the predicted S4 values
        # target are the learned S4 values
        out = (input[0] - target)**2

        # convert weight to a Variable (out is a tensor)
        # (as is done e.g. in binary cross entropy
        # loss here https://github.com/pytorch/pytorch/blob/3bb2308a899e83a9320fdde78e25ae4242251f41/torch/nn/functional.py#L1226 )
        out = out * Variable(self.weight)

        loss = out.sum()
        
        if self.size_average:
            loss /= self.weight.sum()

        return loss

#----------------------------------------------------------------------

from ReLU import ReLU
from Reshape import Reshape

class Model(nn.Module):

    #----------------------------------------

    def __init__(self, rechits_dim):
        super(Model, self).__init__()

        input_size = rechits_dim[0] * rechits_dim[1]

        hidden_size = 2 * input_size

        # network to learn the weights
        # depending on the input shape 
        self.weightsModel = nn.Sequential(
            Reshape(-1, input_size),

            nn.Linear(in_features = input_size, out_features = hidden_size), ReLU(),

            nn.Linear(in_features = hidden_size, out_features = hidden_size), ReLU(),
            nn.Linear(in_features = hidden_size, out_features = hidden_size), ReLU(),
            nn.Linear(in_features = hidden_size, out_features = hidden_size), ReLU(),

            # IMPORTANT: do NOT add a ReLU here after the last layer otherwise 
            # the sigmoid output will always be >= 0.5 !!
            nn.Linear(in_features = hidden_size, out_features = input_size), 

            # in principle we want outputs in the range 0..1 and
            # they should sum to four (not to one !)
            # 
            # but for the moment we use (independent) sigmoids and therefore
            # can have arbitrary shapes
            # 
            # (we could also put a constraint that there is a (2x2) window
            # but let's try not to impose this)
            nn.Sigmoid(),

            # in principle we don't need to reshape
            # back to the original form because
            # in the end we calculate a dot product anyway
            Reshape(-1, rechits_dim[0], rechits_dim[1]),
            )

        # calculate indices of center 5x5 tower
        assert rechits_dim[0] % 2 == 1
        assert rechits_dim[1] % 2 == 1

        # assume python 2 integer division
        center = (rechits_dim[0] / 2, rechits_dim[1] / 2)
        
        self.tower_indices = ( slice(center[0] - 2, center[0] + 3),
                               slice(center[1] - 2, center[1] + 3))
                               
    #----------------------------------------
    def forward(self, x):

        # x is a list 
        xval = x[0]

        # np.set_printoptions(threshold=np.nan,
        #                     precision=3,
        #                     linewidth = 1000)
        

        weights = self.weightsModel.forward(xval)

        # simply multiply the input rechit values 
        # with the predicted weights and sum

        minibatch_size = weights.size(0)

        weighted_sum =  (weights.view(minibatch_size, -1) * xval.view(minibatch_size, -1)).sum(1)

        # divide by sum of center 5x5
        # xval has size  (32L, 1L, 7L, 23L)
        # denominator = x[0][0,0,1:6,9:14].sum

        # tower has size (32,5,5)
        tower = xval[:,0,self.tower_indices[0], self.tower_indices[1]]

        # we sum over dimensions 1 and 2 (keeping the minibatch dimension)
        # we do this in reverse order to avoid shifting the indices
        denominator = tower.sum(dim = 2).sum(dim = 1)

        return [ weighted_sum / denominator,
                 weights
                 ]

#----------------------------------------------------------------------

def makeModel():

    for dim in rechits_dim:
        assert dim % 2 == 1

    return Model(rechits_dim)

#----------------------------------------------------------------------
# function to prepare input data samples
#----------------------------------------------------------------------

unpacker = rechitmodelutils.RecHitsUnpacker(
                    rechits_dim[0], # width
                    rechits_dim[1], # height
                    
                    # for shifting 18,18 to 4,12
                    recHitsXoffset = -18 + rechits_dim[0] / 2 + 1,
                    recHitsYoffset = -18 + rechits_dim[1] / 2 + 1,
                    )

#----------------------------------------------------------------------

import torch.utils.data

class MyDataset(torch.utils.data.Dataset):
    # see also http://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    #----------------------------------------

    def __init__(self, dataset):

        self.weights = dataset['weights']
        self.targets = dataset['phoIdInput/s4'].reshape(-1)

        self.nrows = len(self.weights)

        # unpack rechits here
        self.recHits = unpacker.unpack(dataset, range(self.nrows))


    #----------------------------------------

    def __len__(self):
        return self.nrows

    #----------------------------------------

    def __getitem__(self, index):
        return [ self.weights[index],
                 self.targets[index],
                 [ self.recHits[index] ]  ]

#----------------------------------------------------------------------


def makeDataSet(dataset):
    # note that this is defined in the model file because
    # the dataset we make out of the input files
    # is model dependent

    return MyDataset(dataset)

additionalVars = ['phoIdInput/s4']
normalizeAdditionalVars = False

#----------------------------------------------------------------------

def makeLoss(numOutputNodes, weightsTensor, trainWeights, testWeights):

    trainWeights = trainWeights.reshape((-1,1))
    testWeights  = testWeights.reshape((-1,1))

    # for the moment consider the absolute MSE loss
    # later we could consider taking the MSE loss
    # of the ratio output / target value
    return WeightedMSELoss(weightsTensor, size_average = False), trainWeights, testWeights
