#!/usr/bin/env python

import numpy as np

from lasagne.layers import Conv2DLayer, MaxPool2DLayer, DropoutLayer, ReshapeLayer
from lasagne.nonlinearities import rectify
from lasagne.init import GlorotUniform


#----------------------------------------------------------------------

class RecHitsUnpacker:
    # fills sparse rechits into a tensor

    #----------------------------------------

    def __init__(self, width, height, recHitsXoffset = 0, recHitsYoffset = 0):
        self.width = width
        self.height = height
        self.recHitsXoffset = recHitsXoffset
        self.recHitsYoffset = recHitsYoffset

    #----------------------------------------

    def unpack(self, dataset, rowIndices):
        batchSize = len(rowIndices)

        recHits = np.zeros((batchSize, 1, self.width, self.height), dtype = 'float32')

        # get pointers to these here in order to avoid string comparisons
        # in the loops
        indexOffsets = dataset['rechits']['firstIndex']
        numRecHits   = dataset['rechits']['numRecHits']
        rechitsX     = dataset['rechits']['x']
        rechitsY     = dataset['rechits']['y']
        rechitsE     = dataset['rechits']['energy']

        for i in range(batchSize):

            rowIndex = rowIndices[i]

            # we do NOT subtract one because from 'firstIndex' because
            # these have been already converted in the class SparseConcatenator
            # in datasetutils.py
            indexOffset = indexOffsets[rowIndex]

            nrec = numRecHits[rowIndex]

            # we subtract -1 from the coordinates because these are one based
            # coordinates for Torch (and SparseConcatenator does NOT subtract this)
            xx = rechitsX[indexOffset:indexOffset+nrec] - 1 + self.recHitsXoffset
            yy = rechitsY[indexOffset:indexOffset+nrec] - 1 + self.recHitsYoffset

            selected = (xx >= 0) & (xx < self.width) & (yy >= 0) & (yy < self.height)

            # note that in principle we may have more than one rechit per cell (e.g. in 
            # different bunch crossings)
            recHits[i, 0, xx[selected], yy[selected]] = rechitsE[indexOffset:indexOffset+nrec][selected]

            # end of loop over rechits of this photon
        # end of loop over minibatch indices
        
        return recHits

#----------------------------------------------------------------------

def makeRecHitsModel(network, nstates, filtsize, poolsize):
    # a typical modern convolution network (conv+relu+pool)

    # see https://github.com/torch/nn/blob/master/doc/convolution.md#nn.SpatialModules
    # stage 1 : filter bank -> squashing -> L2 pooling -> normalization

    ### recHitsModel:add(nn.SpatialConvolutionMM(nfeats,             -- nInputPlane
    ###                                   nstates[1],         -- nOutputPlane
    ###                                   filtsize,           -- kernel width
    ###                                   filtsize,           -- kernel height
    ###                                   1,                  -- horizontal step size
    ###                                   1,                  -- vertical step size
    ###                                   (filtsize - 1) / 2, -- padW
    ###                                   (filtsize - 1) / 2 -- padH
    ###                             ))
    ### recHitsModel:add(nn.ReLU())
    
    network = Conv2DLayer(
        network, 
        num_filters = nstates[0], 
        filter_size = (filtsize, filtsize),
        nonlinearity = rectify,
        pad = 'same',
        W = GlorotUniform(),
        )

    # see https://github.com/torch/nn/blob/master/doc/convolution.md#nn.SpatialMaxPooling
    # recHitsModel:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

    network = MaxPool2DLayer(network, pool_size = (poolsize, poolsize),
                             pad = ((poolsize - 1) / 2, (poolsize - 1) / 2)
                             )

    # stage 2 : filter bank -> squashing -> L2 pooling -> normalization
    ### recHitsModel:add(nn.SpatialConvolutionMM(nstates[1],         -- nInputPlane
    ###                                   nstates[2],         -- nOutputPlane
    ###                                   3,                  -- kernel width
    ###                                   3,                  -- kernel height
    ###                                   1,                  -- horizontal step size
    ###                                   1,                  -- vertical step size
    ###                                   (3 - 1) / 2, -- padW
    ###                                   (3 - 1) / 2 -- padH
    ###                             ))
    ### recHitsModel:add(nn.ReLU())

    network = Conv2DLayer(
        network, 
        num_filters = nstates[1], 
        filter_size = (3, 3),
        nonlinearity = rectify,
        pad = 'same',
        W = GlorotUniform(),
        )

    ### recHitsModel:add(nn.SpatialMaxPooling(poolsize, -- kernel width
    ###                                poolsize, -- kernel height
    ###                                poolsize, -- dW step size in the width (horizontal) dimension 
    ###                                poolsize,  -- dH step size in the height (vertical) dimension
    ###                                (poolsize - 1) / 2, -- pad size
    ###                                (poolsize - 1) / 2 -- pad size
    ###                          ))

    network = MaxPool2DLayer(network, pool_size = (poolsize, poolsize),
                             pad = ((poolsize - 1) / 2, (poolsize - 1) / 2)
                             )

    # stage 3 : standard 2-layer neural network
    lastMaxPoolOutputShape = network.output_shape
    print "last maxpool layer output:", lastMaxPoolOutputShape

    # see https://github.com/torch/nn/blob/master/doc/simple.md#nn.View
    # recHitsModel:add(nn.View(nstates[2]*1*5))
    network = ReshapeLayer(network,
                           shape = (-1,              # minibatch dimension
                                     lastMaxPoolOutputShape[-3] * # number of filters
                                     lastMaxPoolOutputShape[-2] * # width
                                     lastMaxPoolOutputShape[-1]   # height
                                     )
                           )

    # recHitsModel:add(nn.Dropout(0.5))
    # it looks like Lasagne scales the inputs at training time
    # while Torch scales them at inference time ?
    network = DropoutLayer(network, p = 0.5)

    return network

#----------------------------------------------------------------------

