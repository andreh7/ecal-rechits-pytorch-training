# datasets including chose and worst track isolation vertex indices (2017-06-23)

# copied from dataset09-tracks-rechits-vtx.py (torch)
# and combined with newer dataset16-rechits.py

datasetDir = '../data/2017-06-23-vertex-info/'

isBarrel = True

dataDesc = dict(
    train_files = [ datasetDir + '/GJet20to40_rechits-barrel-train.npz',
                    datasetDir + '/GJet20toInf_rechits-barrel-train.npz',
                    datasetDir + '/GJet40toInf_rechits-barrel-train.npz',
                    ],

    test_files  = [ datasetDir + '/GJet20to40_rechits-barrel-test.npz',
                    datasetDir + '/GJet20toInf_rechits-barrel-test.npz',
                    datasetDir + '/GJet40toInf_rechits-barrel-test.npz'
                    ],

    inputDataIsSparse = True,

    # if one specifies nothing (or nil), the full sizes
    # from the input samples are taken
    # 
    # if one specifies values < 1 these are interpreted
    # as fractions of the sample
    # trsize, tesize = 10000, 1000
    # trsize, tesize = 0.1, 0.1
    # trsize, tesize = 0.01, 0.01
    
    # limiting the size for the moment because
    # with the full set we ran out of memory after training
    # on the first epoch
    # trsize = 0.5,  tesize = 0.5,

    trsize = None, tesize = None,

    # trsize, tesize = 0.05, 0.05

    # trsize, tesize = 100, 100

    # DEBUG
    # trsize, tesize = 0.01, 0.01
)
#----------------------------------------

doPtEtaReweighting = True

# global variable which can be modified from the command line
additionalVars = []

#----------------------------------------

def datasetLoadFunction(fnames, size, cuda, isTraining, reweightPtEta, logStreams, returnEventIds, 
                                auxData):

    assert not returnEventIds, "not yet supported"

    from datasetutils import getActualSize
    from datasetutilsnpy import makeTracksConcatenator, CommonDataConcatenator, SimpleVariableConcatenator, PtEtaReweighter

    data = None

    totsize = 0

    commonData = CommonDataConcatenator()
    tracks = makeTracksConcatenator([ 'pt', 'charge', 'phiAtVertex', 'etaAtVertex' ] + 
                                    [ 'vtxIndex'   # to find if tracks are associated to the selected or worst index etc.
                                      ])
    photonVars = SimpleVariableConcatenator([
             'phoVars/phoVertexIndex',
             'phoVars/phoSecondWorstIsoVertexIndex',
             'phoVars/phoWorstIsoVertexIndex',

             'phoVars/maxRecHitEta',
             'phoVars/maxRecHitPhi',
             ])

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
        # combine tracks
        #----------
        tracks.add(loaded, thisSize)

        #----------
        # photon variables
        #----------
        photonVars.add(loaded, thisSize)

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
    commonData.normalizeWeights()

    #----------
    data = commonData.data
  
    data['tracks'] = tracks.data

    #----------
    # add photon variables
    #----------
    for key, value in photonVars.data.items():
        assert not key in data
        data[key] = value[:,0]

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

    #----------

    assert totsize == data['tracks']['numTracks'].shape[0]

    return data, totsize

#----------------------------------------------------------------------
