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

    singleVariableInputLayers = []

    from Identity import Identity
    from TableModule import TableModule

    for varname in additionalVars:
        singleVariableInputLayers.append(Identity())

    #----------
    # combine nn output from convolutional layers for
    # rechits with track isolation variables
    #----------
    layers = []

    layers.append(TableModule( [ recHitsModel ] + singleVariableInputLayers ))

    #----------
    # common output part
    #----------

    # to print the shape at this point
    # layers.append(PrintLayer())

    layers.append(nn.Linear(332, nstates[2]))
    nn.init.xavier_uniform(layers[-1].weight.data)

    layers.append(nn.ReLU())


    # output
    layers.append(nn.Linear(nstates[2], 1))
    nn.init.xavier_uniform(layers[-1].weight.data)

    # TODO: we could use logits instead (also requires
    #       different loss function)

    layers.append(nn.Sigmoid())

    result = nn.Sequential(*layers)

    from IndexMerger import IndexMerger

    return IndexMerger(result)

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

def makeInput(dataset, rowIndices, inputDataIsSparse):
    assert inputDataIsSparse,"non-sparse input data is not supported"

    #----------
    # unpack the sparse data
    #----------
    recHits = unpacker.unpack(dataset, rowIndices)

    return [ recHits ] + [
        dataset[varname][rowIndices] for varname in additionalVars
        ]

#----------------------------------------------------------------------
