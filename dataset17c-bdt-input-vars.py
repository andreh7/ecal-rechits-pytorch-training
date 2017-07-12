#!/usr/bin/env python

# datasets with BDT input variables
# copied from ../lasagne-training/dataset14-bdt-inputvars.py
# but with input directory from dataset17b-tracks-only.py

import math
import numpy as np

datasetDir = '../data/2017-06-23-vertex-info/'

dataDesc = dict(

    train_files = [ datasetDir + '/GJet20to40_rechits-barrel-train.npz',
                    datasetDir + '/GJet20toInf_rechits-barrel-train.npz',
                    datasetDir + '/GJet40toInf_rechits-barrel-train.npz',
                    ],

    test_files  = [ datasetDir + '/GJet20to40_rechits-barrel-test.npz',
                    datasetDir + '/GJet20toInf_rechits-barrel-test.npz',
                    datasetDir + '/GJet40toInf_rechits-barrel-test.npz',
                    ],

    inputDataIsSparse = False,

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
#----------------------------------------


### -- this is called after loading and combining the given
### -- input files
### function postLoadDataset(label, dataset)
### 
### end

#--------------------------------------
# input variables
#--------------------------------------
#
#   phoIdInput :
#     {
#       covIEtaIEta : FloatTensor - size: 431989          0
#       covIEtaIPhi : FloatTensor - size: 431989          1
#       esEffSigmaRR : FloatTensor - size: 431989         2 # endcap only
#       etaWidth : FloatTensor - size: 431989             3
#       pfChgIso03 : FloatTensor - size: 431989           4
#       pfChgIso03worst : FloatTensor - size: 431989      5
#       pfPhoIso03 : FloatTensor - size: 431989           6
#       phiWidth : FloatTensor - size: 431989             7
#       r9 : FloatTensor - size: 431989                   8
#       rho : FloatTensor - size: 431989                  9
#       s4 : FloatTensor - size: 431989                  10
#       scEta : FloatTensor - size: 431989               11
#       scRawE : FloatTensor - size: 431989              12
#     }

# names of variables (without the preceding 'phoIdInput/') for
# which we should take the absolute value
absVarNames = [
 'covIEtaIPhi', # not sure whether this is symmetric by design ?
 'scEta',
    ]

# by default, normalize input variables
# (can be overridden on the command line)
normalizeBDTvars = True

#----------------------------------------------------------------------

def datasetLoadFunction(fnames, size, cuda, isTraining, reweightPtEta, logStreams, returnEventIds, auxData):
    # @param returnEventIds if True, returns also a dict with sample/run/ls/event numbers 

    # only apply pt/eta reweighting for training dataset
    reweightPtEta = reweightPtEta and isTraining

    from datasetutils import getActualSize
    from datasetutilsnpy import CommonDataConcatenator, SimpleVariableConcatenator, SimpleVariableConcatenatorToMatrix, PtEtaReweighter

    #----------
    if reweightPtEta:
      # for reweighting (use reconstructed pt and eta)
      ptEta = SimpleVariableConcatenator(['pt', 'eta'],
                                         dict(pt = lambda loaded:  loaded['phoVars/phoEt'],
                                              eta = lambda loaded: loaded['phoIdInput/scEta'])
                                         )

    #----------
    data = None

    totsize = 0

    commonData = CommonDataConcatenator()

    bdtVars = None 
                                                 
    # sort the names of the input variables
    # so that we get reproducible results
    sortedVarnames = {}

    if returnEventIds:
        eventIDs = {}
  
    # load all input files
    for fname in fnames:
  
        print "reading",fname
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
        # BDT input variables
        #----------
        if bdtVars == None:
            # find the variable names
            groupVarName = 'phoIdInput'

            # fill the individual variable names
            sortedVarnames = sorted([ key for key in loaded.keys() if key.startswith(groupVarName + "/")])

            # for barrel, exclude the preshower variable
            # (this will have zero standard deviation in the barrel and will
            # therefore lead to NaNs after normalization)
            sortedVarnames = [ varname for varname in sortedVarnames if varname != groupVarName + '/esEffSigmaRR' ]

            # filter on selected variables if specified
            from datasetutilsnpy import selectVariables
            
            sortedVarnames = selectVariables(sortedVarnames, selectedVariables)
            assert sortedVarnames, "no variables left after applying selectedVariables"

            bdtVars = SimpleVariableConcatenatorToMatrix(groupVarName, sortedVarnames)

        bdtVars.add(loaded, thisSize)

        #----------
        # pt/eta reweighting variables
        #----------
        if reweightPtEta:
          ptEta.add(loaded, thisSize)

        #----------
        # run/event etc.
        #----------
        if returnEventIds:
            for key in ('sample', 'run', 'ls', 'event'):
                if eventIDs.has_key(key):
                    eventIDs[key] = np.concatenate( [ eventIDs[key], loaded[key] ])
                else:
                    eventIDs[key] = loaded[key]

        #----------
        # encourage garbage collection
        #----------
        del loaded

    # end of loop over input files

    # take absolute value of some variables
    for col,varname in enumerate(sortedVarnames):
        if varname.startswith(groupVarName + "/"):
            varname = varname[(len(groupVarName) + 1):]

        if varname in absVarNames:
            for fout in logStreams:
                print >> fout, "taking absolute value of",varname
            bdtVars.makeAbsValue(col)

    if normalizeBDTvars:
        bdtVars.normalize()

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
    commonData.normalizeWeights()
        
    # combine the datas
    data = commonData.data
    data['input'] = bdtVars.data

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


    assert totsize == data['input'].shape[0]

    if returnEventIds:
        return data, totsize, eventIDs
    else:
        return data, totsize

#----------------------------------------------------------------------

selectedVariables = None
