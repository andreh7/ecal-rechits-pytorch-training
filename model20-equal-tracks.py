#!/usr/bin/env python

# model using tracks only in a way that individual
# track information is used and no hierarchy or
# ordering is imposed on the tracks

#----------------------------------------------------------------------
# (default) model parameters
#----------------------------------------------------------------------

# size and number of hidden layers on input side

#----------------------------------------------------------------------
# function to prepare input data samples
#----------------------------------------------------------------------

def makeInput(dataset, rowIndices, inputDataIsSparse):
    assert inputDataIsSparse,"non-sparse input data is not supported"

    # maximum distance (in cm) for which a vertex is considered to be
    # 'the same' or not
    maxVertexDist = 0.01 # 100 um

    detaDphiFunc = lambda dataset, photonIndex, trackIndex: (
        dataset['tracks']['etaAtVertex'][trackIndex] - dataset['phoVars/maxRecHitEta'][photonIndex],
        dataset['tracks']['phiAtVertex'][trackIndex] - dataset['phoVars/maxRecHitPhi'][photonIndex],
        )

    retval = [ 
        ]

    for trackFilter in (
        ### # tracks from same vertex as diphoton candidate
        ### lambda dataset, photonIndex, trackIndex: dataset['tracks']['vtxIndex'][trackIndex] == dataset['phoVars/phoVertexIndex'][photonIndex],

        # tracks from the worst iso vertex 
        lambda dataset, photonIndex, trackIndex: dataset['tracks']['vtxIndex'][trackIndex] == dataset['phoVars/phoWorstIsoVertexIndex'][photonIndex],

        ### # tracks from the second worst iso vertex
        ### lambda dataset, photonIndex, trackIndex: dataset['tracks']['vtxIndex'][trackIndex] == dataset['phoVars/phoSecondWorstIsoVertexIndex'][photonIndex],
        ### 
        ### # tracks from other vertices
        ### lambda dataset, photonIndex, trackIndex: dataset['tracks']['vtxIndex'][trackIndex] != dataset['phoVars/phoVertexIndex'][photonIndex] and \
        ###                                          dataset['tracks']['vtxIndex'][trackIndex] != dataset['phoVars/phoWorstIsoVertexIndex'][photonIndex] and \
        ###                                          dataset['tracks']['vtxIndex'][trackIndex] != dataset['phoVars/phoSecondWorstIsoVertexIndex'][photonIndex],

        ):

        # build list of list of tracks from the given vertex
        # for the moment just use the trackPt
        trackFirstIndex = dataset['tracks']['firstIndex']
        numTracks = dataset['tracks']['numTracks']

        trackPt = dataset['tracks']['pt']

        numVarsPerTrack = 1

        thisVertexValues = []

        for photonIndex in rowIndices:
            
            thisVal = np.zeros( (numTracks[photonIndex], numVarsPerTrack) , dtype='float32')
            trackIndexOffset = trackFirstIndex[photonIndex]
  
            # assign variables vectorized
            thisVal[:,0] = trackPt[trackIndexOffset:trackIndexOffset + numTracks[photonIndex]]

            thisVertexValues.append(thisVal)
        retval.append(thisVertexValues)

    # end of loop over vertex types

    # first index is vertex type
    # second index is photon index
    # value is 2D array of tracks (size varies from event to event)

    return retval

#----------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import operator

class Net(nn.Module):
    
    def __init__(self, 
                 numLayersInputSide = 5, 
                 widthInputSide = 50,
                 numLayersCommonSide = 5,
                 widthCommonSide = 50,
                ):
        
        super(Net, self).__init__()
        
        #----------
        # input side layers
        #----------

        numInputs = 1
        
        self.inputSideLayers = []
        for i in range(numLayersInputSide):
            layer = nn.Linear(numInputs, widthInputSide)
            self.inputSideLayers.append(layer)
            self.add_module("iLayer%d" % i, layer)
            
            numInputs = widthInputSide

        #----------
        # output side layers
        #----------

        numInputs = widthInputSide
        numOutputs = widthCommonSide
        
        self.commonSideLayers = []
        for i in range(numLayersCommonSide):
          
            if i == numLayersCommonSide - 1:
                numOutputs = 1
            else:
                numOutputs = widthCommonSide
            
            layer = nn.Linear(numInputs, numOutputs)
            self.commonSideLayers.append(layer)
            self.add_module("oLayer%d" % i, layer)
            
            numInputs = numOutputs

        # neutral element as input to output network
        # for rows with no tracks
        self.noTracksIntermediateOutput = Variable(torch.zeros(1,widthCommonSide))
        if cuda:
            self.noTracksIntermediateOutput = self.noTracksIntermediateOutput.cuda()

    #----------------------------------------
    
    def forward(self, dataset, indices):

        # we only have one vertex
        thisVtxData = dataset[0]
        
        # for each row, 
        #   - feed this average to the output side network
        
        # overall output for the entire minibatch
        outputs = []
        
        # loop over minibatch entries
        for index in indices:

            # input is a 2D tensor: 
            #   first index is the index of the track within the row
            #   second index is the variable index

            numPoints = thisVtxData[index].shape[0]
            numVars   = thisVtxData[index].shape[1]

            if numPoints > 0:
                h = Variable(torch.from_numpy(thisVtxData[index]))

                if cuda:
                    h = h.cuda()

                # forward all input points through the input side network
                for layer in self.inputSideLayers:
                    h = layer(h)
                    h = F.relu(h)

                # average the input side network outputs: sum along first dimension (point index), 
                # then divide by number of points
                output = h.sum(0) / numPoints
            else:
                # no tracks in this event
                output = self.noTracksIntermediateOutput
            
            # feed through the output side network
            h = output
            for layerIndex, layer in enumerate(self.commonSideLayers):
                
                h = layer(h)

                if layerIndex == len(self.commonSideLayers) - 1:
                    # apply sigmoid at output of last layer
                    h = F.sigmoid(h)
                else:
                    h = F.relu(h)
                
            outputs.append(h)
            
        # end of loop over minibatch entries
         
        # convert the list of outputs to a torch 2D tensor
        return torch.cat(outputs, 0)[:,0]
            


#----------------------------------------------------------------------

def makeModel():

    return Net()

#----------------------------------------------------------------------

