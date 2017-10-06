#!/usr/bin/env python

# like model06 but with dropout layer only applied
# to the rechits variables, not the other (track iso)
# variables


from lasagne.layers import InputLayer, DenseLayer, ConcatLayer
from lasagne.init import GlorotUniform
from lasagne.nonlinearities import rectify, sigmoid

import rechitmodelutils

import numpy as np
import theano.tensor as T
import math

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

    # note that we need several input variables here
    # 3D tensor
    inputVarRecHits        = T.tensor4('rechits')

    ninputLayers = 1
    network = InputLayer(shape=(None, ninputLayers, width, height),
                         input_var = inputVarRecHits,
                         name = 'rechits',
                        )

    recHitsModel = rechitmodelutils.makeRecHitsModel(network, nstates[:2], filtsize, poolsize)

    # create input 'networks' for additional variables (such as track isolation variables)
    # taking the global variable additionalVars specified in a dataset file or modified
    # on the command line
    singleVariableInputVars = []    # Theano variables
    singleVariableInputLayers = []  # Lasagne input layers

    # TODO: could use a single input layer for the additional variables
    #       instead of individual ones

    for varname in additionalVars:
        singleVariableInputVars.append(T.matrix('varname'))

        singleVariableInputLayers.append(
            InputLayer(shape = (None,1), 
                       input_var = singleVariableInputVars[-1],
                       name = varname)
            )

    #----------
    # combine nn output from convolutional layers for
    # rechits with track isolation variables
    #----------

    network = ConcatLayer([ recHitsModel ] + singleVariableInputLayers,
                          axis = 1)

    #----------
    # common output part
    #----------
    # outputModel:add(nn.Linear(nstates[2]*1*5 + 2, # +2 for the track isolation variables
    #                           nstates[3]))
    # outputModel:add(nn.ReLU())

    network = DenseLayer(
        network,
        num_units = nstates[2],
        W = GlorotUniform(),
        nonlinearity = rectify)

    # output
    network = DenseLayer(
        network,
        num_units = 1,  
        nonlinearity = sigmoid,
        W = GlorotUniform(),
        )

    return [ inputVarRecHits ] + singleVariableInputVars, network

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
