#!/usr/bin/env python

# like model06 but with dropout layer only applied
# to the rechits variables, not the other (track iso)
# variables


import rechitmodelutils

import numpy as np
import math

import torch.nn as nn
import torch.nn.functional as F

#----------------------------------------------------------------------
# model
#----------------------------------------------------------------------

# 2-class problem
noutputs = 2

# input dimensions
width = 7
height = 23

ninputs = 1 * width * height

# hidden units, filter sizes for convolutional network
nstates = [64,64,128]
filtsize = 5
poolsize = 2

#----------------------------------------

# size of minibatch
batchSize = 32

# how many minibatches to unpack at a time
# and to store in the GPU (to have fewer
# data transfers to the GPU)
batchesPerSuperBatch = math.floor(3345197 / batchSize)

#----------------------------------------------------------------------

def makeModel():

    # recHitsModel will be an nn.Sequential module
    recHitsModel = rechitmodelutils.makeRecHitsModel(nstates[:2], filtsize, poolsize)

    # create input 'networks' for additional variables (such as track isolation variables)
    # taking the global variable additionalVars specified in a dataset file or modified
    # on the command line

    # TODO: could use a single input layer for the additional variables
    #       instead of individual ones

    from Identity import Identity
    from TableModule import TableModule
    from JoinTable import JoinTable

    #----------
    # combine nn output from convolutional layers for
    # rechits with track isolation variables
    #----------
    layers = []

    layers.append(TableModule( [ recHitsModel, Identity() ]  ))
    layers.append(JoinTable(1))

    #----------
    # common output part
    #----------

    # to print the shape at this point
    # layers.append(PrintLayer())

    layers.append(nn.Linear(320 + len(additionalVars), nstates[2]))
    nn.init.xavier_uniform(layers[-1].weight.data)

    layers.append(nn.ReLU())


    # output
    layers.append(nn.Linear(nstates[2], 1))
    nn.init.xavier_uniform(layers[-1].weight.data)

    # TODO: we could use logits instead (also requires
    #       different loss function)

    layers.append(nn.Sigmoid())

    result = nn.Sequential(*layers)

    return result

#----------------------------------------------------------------------
# function to prepare input data samples
#----------------------------------------------------------------------

unpacker = rechitmodelutils.RecHitsUnpacker(
    width,
    height,
    # for shifting 18,18 to 4,12

    recHitsXoffset = -18 + 4,
    recHitsYoffset = -18 + 12,
    )

#----------------------------------------------------------------------

import torch.utils.data

class MyDataset(torch.utils.data.Dataset):
    # see also http://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    #----------------------------------------

    def __init__(self, dataset, additionalVarNames):

        self.weights = dataset['weights']
        self.targets = dataset['labels']

        self.nrows = len(self.weights)

        # unpack rechits here
        self.recHits = unpacker.unpack(dataset, range(self.nrows))

        # unpack additionalVars into a 2D matrix
        self.additionalVars = np.stack([ dataset[varname] for varname in additionalVarNames], axis = 1).squeeze()# .ravel()# .reshape(-1, len(additionalVars))[rowIndices] ]

    #----------------------------------------

    def __len__(self):
        return self.nrows

    #----------------------------------------

    def __getitem__(self, index):
        return [ self.weights[index],
                 self.targets[index],
                 self.recHits[index], self.additionalVars[index] ]

#----------------------------------------------------------------------


def makeDataSet(dataset):
    # note that this is defined in the model file because
    # the dataset we make out of the input files
    # is model dependent

    return MyDataset(dataset, additionalVars)
