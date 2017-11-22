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

        batch_size = input.size(0)

        out = (input.view(batch_size,-1) - target.view(batch_size,-1))**2

        out = out.sum(1)

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

            nn.Linear(in_features = hidden_size, out_features = input_size), ReLU(),

            # we want outputs in the range 0..1 and
            # they should sum to one
            nn.Softmax(),

            # in principle we don't need to reshape
            # back to the original form because
            # in the end we calculate a dot product anyway
            Reshape(-1, rechits_dim[0], rechits_dim[1]),
            )

    #----------------------------------------
    def forward(self, x):

        # x is a list 
        xval = x[0]

        # np.set_printoptions(threshold=np.nan,
        #                     precision=3,
        #                     linewidth = 1000)
        

        weights = self.weightsModel.forward(xval)

        return weights

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
        self.targets = dataset['phoIdInput/s4']

        self.nrows = len(self.weights)

        # unpack rechits here
        self.recHits = unpacker.unpack(dataset, range(self.nrows))

        # find the four (adjacent) cells
        # giving S4

        conv_weights = np.ones((2,2))
        
        # find the coordinate of the center rechit
        rechits_mid = np.array([rechits_dim[0] / 2, rechits_dim[1] / 2])
        towers = self.recHits[:,0,
                                (rechits_mid[0]-2):(rechits_mid[0]+3),
                                (rechits_mid[1]-2):(rechits_mid[1]+3),
                                ]
        self.masks = np.zeros((len(towers), rechits_dim[0], rechits_dim[1]), dtype = 'float32')

        # we need to shift back from (5x5) maximum to the 7x23 window
        #                     center of 5x5
        shift = rechits_mid - np.array([2,2])

        # find the windows with the maximum 2x2 sum
        import scipy.signal
        for i in range(len(towers)):
            conv = scipy.signal.convolve2d(towers[i], conv_weights, mode = 'valid')

            # the maximum coordinate is the top left of the window
            # note that in some cases the window is not unique
            # (e.g. a single value surrounded by zeros)
            maxpos = np.unravel_index(conv.argmax(), conv.shape)

            # maxpos comes from the 5x5 tower, we need to shift it
            # back to the 7x23 image
            maxpos = np.array(maxpos) + shift

            self.masks[i,
                  maxpos[0]:(maxpos[0]+2),
                  maxpos[1]:(maxpos[1]+2)] = 1

            # DEBUG
            if i < 0:
                np.set_printoptions(linewidth = 300, suppress = True, precision = 2)
                print "tower="; print towers[i]
                print "mask="; print self.masks[i]
                print "s4 cacluated=", np.dot(recHits[i].ravel(), masks[i].ravel()) / towers[i].sum()
            

    #----------------------------------------

    def __len__(self):
        return self.nrows

    #----------------------------------------

    def __getitem__(self, index):
        return [ self.weights[index],
                 self.masks[index],    # target to learn
                 self.recHits[index] ]

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
