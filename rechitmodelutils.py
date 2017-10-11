#!/usr/bin/env python

import numpy as np
import torch.nn as nn

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

def makeRecHitsModel(nstates, filtsize, poolsize):
    # a typical modern convolution network (conv+relu+pool)

    # see https://github.com/torch/nn/blob/master/doc/convolution.md#nn.SpatialModules

    #----------
    # stage 1 : filter bank -> squashing -> L2 pooling -> normalization
    #----------

    layers = []

    ### recHitsModel:add(nn.SpatialConvolutionMM(nfeats,             -- nInputPlane
    ###                                   nstates[1],         -- nOutputPlane
    ###                                   filtsize,           -- kernel width
    ###                                   filtsize,           -- kernel height
    ###                                   1,                  -- horizontal step size
    ###                                   1,                  -- vertical step size
    ###                                   (filtsize - 1) / 2, -- padW
    ###                                   (filtsize - 1) / 2 -- padH
    ###                             ))

    layers.append(nn.Conv2d(
            in_channels  = 1,
            out_channels = nstates[0],
            kernel_size  = (filtsize, filtsize),
            padding      = ((filtsize - 1) / 2, (filtsize - 1) / 2),
            ))

    nn.init.xavier_uniform(layers[-1].weight.data)

    layers.append(nn.ReLU())

    layers.append(nn.MaxPool2d(
            kernel_size = (poolsize, poolsize),
            stride      = (poolsize, poolsize),
            padding     = ((poolsize - 1) / 2, (poolsize - 1)/2),
            ))

    #----------
    # stage 2 : filter bank -> squashing -> L2 pooling -> normalization
    #----------
    ### recHitsModel:add(nn.SpatialConvolutionMM(nstates[1],         -- nInputPlane
    ###                                   nstates[2],         -- nOutputPlane
    ###                                   3,                  -- kernel width
    ###                                   3,                  -- kernel height
    ###                                   1,                  -- horizontal step size
    ###                                   1,                  -- vertical step size
    ###                                   (3 - 1) / 2, -- padW
    ###                                   (3 - 1) / 2 -- padH
    ###                             ))

    layers.append(nn.Conv2d(
            in_channels  = nstates[0],
            out_channels = nstates[1],
            kernel_size  = (3, 3),
            padding      = ((3 - 1) / 2, (3 - 1) / 2),
            ))

    nn.init.xavier_uniform(layers[-1].weight.data)

    layers.append(nn.ReLU())

    ### recHitsModel:add(nn.SpatialMaxPooling(poolsize, -- kernel width
    ###                                poolsize, -- kernel height
    ###                                poolsize, -- dW step size in the width (horizontal) dimension 
    ###                                poolsize,  -- dH step size in the height (vertical) dimension
    ###                                (poolsize - 1) / 2, -- pad size
    ###                                (poolsize - 1) / 2 -- pad size
    ###                          ))

    layers.append(nn.MaxPool2d(
            kernel_size = (poolsize, poolsize),
            stride      = (poolsize, poolsize),
            padding     = ((poolsize - 1) / 2, (poolsize - 1)/2),
            ))
    #----------
    # stage 3 : standard 2-layer neural network
    #----------

    # see https://github.com/torch/nn/blob/master/doc/simple.md#nn.View
    # recHitsModel:add(nn.View(nstates[2]*1*5))

    from FlattenLayer import FlattenLayer

    layers.append(FlattenLayer())

    # TODO: check if we should add the dropout layer
    #       here or after combining with the other variables if any ?
    layers.append(nn.Dropout(p = 0.5))

    return nn.Sequential(*layers)

#----------------------------------------------------------------------

