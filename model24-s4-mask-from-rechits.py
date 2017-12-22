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

# rechits_dim = (7,23)
rechits_dim = (5,5)

add_s4_estimates = True

#----------------------------------------------------------------------
# model
#----------------------------------------------------------------------

# BCE loss with some reshaping
class MyLoss(torch.nn.modules.loss._WeightedLoss):

    ### def __init__(self, weights_tensor):
    ###     super(MyLoss, self).__init__(weights_tensor)
    ### 
    ###     # expects numerical labels, not one hot encoded labels
    ###     # see also https://discuss.pytorch.org/t/feature-request-nllloss-crossentropyloss-that-accepts-one-hot-target/2724
    ###     # self.loss = nn.CrossEntropyLoss(weights_tensor, size_average = False)
    ### 
    ###     # does not work either
    ###     # self.loss = nn.MultiLabelSoftMarginLoss(weights_tensor, size_average = False)

    def forward(self, input, target):

        input = input[0]
        
        batch_size = input.size(0)

        # does not work
        # return torch.nn.functional.binary_cross_entropy(input, target, self.weight, size_average = self.size_average)

        torch.nn.modules.loss._assert_no_grad(target)
        
        # calculate the per cell binary cross entropy loss ourselves
        # out = target * torch.log(input) + (1 - target) * torch.log(1 - input)

        loss = 0

        for i in range(batch_size):
             loss += self.weight[i] * torch.nn.functional.binary_cross_entropy(input[i], target[i], size_average = False)

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

    #----------------------------------------
    def forward(self, x):

        # x is a list 
        xval = x[0]

        # np.set_printoptions(threshold=np.nan,
        #                     precision=3,
        #                     linewidth = 1000)
        

        weights = self.weightsModel.forward(xval)

        # also calculate the s4 value from the mask
        # for testing (note that we do NOT include
        # them into the loss)
        if add_s4_estimates:

            minibatch_size = weights.size(0)

            tower = xval[:,0]

            # print "tower=",tower.size(),weights.view(minibatch_size, -1).size(), tower.view(minibatch_size, -1).size()

            weighted_sum = (weights.view(minibatch_size, -1) * tower.view(minibatch_size, -1)).sum(1)

            # we sum over dimensions 1 and 2 (keeping the minibatch dimension)
            # we do this in reverse order to avoid shifting the indices
            denominator = tower.sum(dim = 2).sum(dim = 1)

            s4 = weighted_sum / denominator

            return [ weights, s4 ]
        else:
            return [ weights ]
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

        if add_s4_estimates:
            self.s4 = dataset['phoIdInput/s4']

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
        nrows = len(towers)

        for i in range(nrows):
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

        if add_s4_estimates:
            # also calculate our S4 based on these masks

            numerator = (towers.reshape(nrows, -1) * self.masks.reshape(nrows, -1)).sum(axis = 1)

            denominator = towers.reshape(nrows, -1).sum(axis = 1)

            self.s4recalculated = numerator / denominator

    #----------------------------------------

    def __len__(self):
        return self.nrows

    #----------------------------------------

    def __getitem__(self, index):

        result =  [ self.weights[index],
                    self.masks[index],   # target to learn
                    [ self.recHits[index] ]  # inputs
                    ]

        if add_s4_estimates:
            result.append([ self.s4[index], self.s4recalculated[index] ])

        return result

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
    return MyLoss(weightsTensor), trainWeights, testWeights
