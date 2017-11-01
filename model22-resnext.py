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

def makeModel():

    layers = []
    import resnext
    from ParallelTable import ParallelTable
    from JoinTable import JoinTable

    model = resnext.ModelCreator(depth = 29, 
                                       cardinality = 16,
                                       baseWidth = 64,
                                       
                                       dataset = 'cifar10',
                                       bottleneckType = 'resnext_C',
                                       numInputPlanes = 1,
                                       avgKernelSize = 9, # for 35x35 inputs
                                       numOutputNodes = 1,
                                       ).create()

    layers.append(ParallelTable( [ model ]  ))
    layers.append(JoinTable(1))

    layers.append(nn.Sigmoid())

    result = nn.Sequential(*layers)

    return result

#----------------------------------------------------------------------
# function to prepare input data samples
#----------------------------------------------------------------------

unpacker = rechitmodelutils.RecHitsUnpacker(
    35, # width,
    35, # height,
    # for shifting 18,18 to 4,12

    # recHitsXoffset = -18 + 4,
    # recHitsYoffset = -18 + 12,
    )

#----------------------------------------------------------------------

import torch.utils.data

class MyDataset(torch.utils.data.Dataset):
    # see also http://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    #----------------------------------------

    def __init__(self, dataset):

        self.weights = dataset['weights']
        self.targets = dataset['labels']

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
                 self.recHits[index] ]

#----------------------------------------------------------------------


def makeDataSet(dataset):
    # note that this is defined in the model file because
    # the dataset we make out of the input files
    # is model dependent

    return MyDataset(dataset)
