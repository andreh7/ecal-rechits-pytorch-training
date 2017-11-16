#!/usr/bin/env python

# datasets with BDT input variables

import math
import numpy as np

datasetDir = '../data/2017-03-19-npy-mass-cut'

isBarrel = True

dataDesc = dict(

    train_files = [ datasetDir + '/GJet20to40_rechits-barrel-train.npz',
                    datasetDir + '/GJet20toInf_rechits-barrel-train.npz',
                    datasetDir + '/GJet40toInf_rechits-barrel-train.npz',
                    ],

    test_files  = [ datasetDir + '/GJet20to40_rechits-barrel-test.npz',
                    datasetDir + '/GJet20toInf_rechits-barrel-test.npz',
                    datasetDir + '/GJet40toInf_rechits-barrel-test.npz',
                    ],

    inputDataIsSparse = True,

    # if one specifies nothing (or None), the full sizes
    # from the input samples are taken

    #  if one specifies values < 1 these are interpreted
    #  as fractions of the sample
    #  trsize, tesize = 10000, 1000
    #  trsize, tesize = 0.1, 0.1
    #  trsize, tesize = 0.01, 0.01
    # 
    #  limiting the size for the moment because
    #  with the full set we ran out of memory after training
    #  on the first epoch
    #  trsize, tesize = 0.5, 0.5
    
    trsize = None, tesize = None,


    # DEBUG
    # trsize = 0.01, tesize = 0.01,
    # trsize, tesize = 100, 100
)   

doPtEtaReweighting = True

# global variable which can be modified from the command line
if not globals().has_key('additionalVars'):
    global additionalVars
    additionalVars = []

if not globals().has_key('normalizeAdditionalVars'):
    global noramlizeAdditionalVars
    normalizeAdditionalVars = True

#----------------------------------------------------------------------

def __datasetLoadFunctionHelper(fnames, size, cuda, isTraining, reweightPtEta, logStreams, returnEventIds,
                                auxData,
                                additionalVars = []):
    # @param returnEventIds if True, returns also a dict with sample/run/ls/event numbers 
    # @param additionalVars is a list of 'simple' variables such as track isolation variables to be added as
    #                       a separate group of inputs

    from datasetutils import getActualSize
    from datasetutilsnpy import makeRecHitsConcatenator, CommonDataConcatenator, SimpleVariableConcatenator, PtEtaReweighter

    data = None

    totsize = 0

    commonData = CommonDataConcatenator()
    recHits = makeRecHitsConcatenator()

    # only apply pt/eta reweighting for training dataset
    reweightPtEta = reweightPtEta and isTraining

    #----------
    if reweightPtEta:
      # for reweighting (use reconstructed pt and eta)
      ptEta = SimpleVariableConcatenator(['pt', 'eta'],
                                         dict(pt = lambda loaded:  loaded['phoVars/phoEt'],
                                              eta = lambda loaded: loaded['phoIdInput/scEta'])
                                         )

    #----------
                                                 
    assert not returnEventIds

    # some of the BDT input variables
    if additionalVars:
        otherVars = SimpleVariableConcatenator(additionalVars)

    # load all input files
    for fname in fnames:

        for log in logStreams:
            print >> log, "reading",fname

        loaded = np.load(fname)

        #----------
        # determine the size
        #----------
        thisSize = getActualSize(size, loaded)

        totsize += thisSize

        #----------
        # combine common data
        #----------
        commonData.add(loaded, thisSize)

        #----------
        # combine rechits
        #----------
        recHits.add(loaded, thisSize)
    
        #----------
        # auxiliary variables
        #----------
        if additionalVars:
            otherVars.add(loaded, thisSize)

        #----------
        # pt/eta reweighting variables
        #----------
        if reweightPtEta:
          ptEta.add(loaded, thisSize)

        #----------
        # encourage garbage collection
        #----------
        del loaded

    # end of loop over input files

    #----------
    # reweight signal to have the same background shape
    # using a 2D (pt, eta) histogram
    #----------
    if reweightPtEta:
      ptEtaReweighter = PtEtaReweighter(ptEta.data['pt'][:,0],
                                        ptEta.data['eta'][:,0],
                                        commonData.data['labels'],
                                        isBarrel)

      scaleFactors = ptEtaReweighter.getSignalScaleFactors(ptEta.data['pt'][:,0],
                                                           ptEta.data['eta'][:,0],
                                                           commonData.data['labels'])
      
      # keep original weights
      commonData.data['weightsBeforePtEtaReweighting'] = np.copy(commonData.data['weights'])


      commonData.data['weights'] *= scaleFactors

    #----------
    # normalize event weights
    #----------
    # make average weight equal to one over the sample
    commonData.normalizeWeights()
        
    # combine the datas
    data = commonData.data

    # add rechits
    data['rechits'] = recHits.data

    #----------
    # normalize auxiliary variables to zero mean and unit variance
    #----------
    if additionalVars:
        if normalizeAdditionalVars:
            otherVars.normalize()

        #----------
        # add auxiliary variables
        #----------
        for key, value in otherVars.data.items():
            assert not key in data
            data[key] = value


    #----------
    # cross check for pt/eta reweighting, dump some variables
    #----------
    if reweightPtEta:
      outputName = "/tmp/pt-reweighting.npz"
      np.savez(outputName,
               pt = ptEta.data['pt'][:,0],
               eta = ptEta.data['eta'][:,0],
               weights = commonData.data['weights'],
               labels = commonData.data['labels'],
               scaleFactors = scaleFactors,
               sigHistogram = ptEtaReweighter.sigHistogram,
               bgHistogram = ptEtaReweighter.bgHistogram,
               )
      print "wrote pt/eta reweighting data to", outputName


    assert totsize == data['rechits']['numRecHits'].shape[0]

    return data, totsize

#----------------------------------------------------------------------


def datasetLoadFunction(fnames, size, cuda, isTraining, reweightPtEta, logStreams, returnEventIds,
                        auxData):
    return __datasetLoadFunctionHelper(fnames, size, cuda, isTraining, reweightPtEta, logStreams, 
                                       returnEventIds, auxData,
                                       additionalVars = additionalVars)
