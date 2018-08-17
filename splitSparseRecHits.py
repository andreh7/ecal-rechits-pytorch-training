#!/usr/bin/env python

# splits a rechits file into test and train file

import sys
import numpy as np
import fnmatch

#----------------------------------------------------------------------

# fraction of test sample
testFraction = 0.25

randSeed = 1337

#----------------------------------------------------------------------

# @param ignoreKeys is a list of fnmatch patterns
def makeOutputVar(inputData, indices, ignoreKeys):
    # inputData must be a dict
  
    outputData = {}
  
    for key, value in inputData.items():
      
        # check if we should not touch this key
        if any(fnmatch.fnmatch(key, pattern) for pattern in ignoreKeys):
            continue

        # note that in the npy version value is never a dict 
        # but a numpy object

        outputData[key] = value[indices]

    # end of loop over items in the table
  
    return outputData

#----------------------------------------------------------------------

def makeOutputDataRecHits(indices, inputData, outputData):

    numOutputRows = len(indices)
  
    #----------
    # calculate the total number of output rechits
    #----------
    numOutputRecHits = inputData['X/numRecHits'][indices].sum()
  
    #----------
  
    # now that we know the total number of output rechits, 
    # copy the vectors related to the rechits 

    outputData['X/x']      = -1 * np.ones(numOutputRecHits, dtype = 'int32')
    outputData['X/y']      = -1 * np.ones(numOutputRecHits, dtype = 'int32')
    outputData['X/energy'] = -1 * np.ones(numOutputRecHits, dtype = 'float32')
  
    outputData['X/firstIndex'] = -1 * np.ones(numOutputRows, dtype = 'int32')
    outputData['X/numRecHits'] = inputData['X/numRecHits'][indices]
  
    # note that we keep the one based convention from Torch here for
    # historical reasons
    firstIndex = 1

    import tqdm
    progbar = tqdm.tqdm(total = numOutputRows, 
                        mininterval = 0.1, 
                        unit = 'photons',
                        desc = 'splitting rechits')
  

    # assign to variables to avoid dict lookups all the time (which makes it very slow, at least for npz files...)

    inputFirstIndex = inputData['X/firstIndex']
    inputNumRecHits = inputData['X/numRecHits']
    inputDataX = inputData['X/x']     
    inputDataY = inputData['X/y']     
    inputDataE = inputData['X/energy']

    outputDataX = outputData['X/x']
    outputDataY = outputData['X/y']
    outputDataE = outputData['X/energy']
    outputDataFirstIndex = outputData['X/firstIndex']
    outputDataNumRecHits = outputData['X/numRecHits']

    for i in range(numOutputRows):

        # this is zero based
        index = indices[i]
    
        outputDataFirstIndex[i] = firstIndex
    
        # sanity check of input data
        assert inputFirstIndex[index] >= 1
        assert inputFirstIndex[index] + inputNumRecHits[index] - 1 <= len(inputDataE), "failed at index=" + str(index)
    
        # baseInputIndex is zero based
        baseInputIndex = inputFirstIndex[index] - 1

        thisNumRecHits = inputNumRecHits[index]
        outputDataNumRecHits[i] = thisNumRecHits

        # copy coordinates and energy over
        # note that firstIndex is one based
        

        outputDataX[(firstIndex - 1):(firstIndex - 1 + thisNumRecHits)] = inputDataX[(baseInputIndex):(baseInputIndex + thisNumRecHits)] 
        outputDataY[(firstIndex - 1):(firstIndex - 1 + thisNumRecHits)] = inputDataY[(baseInputIndex):(baseInputIndex + thisNumRecHits)] 
        outputDataE[(firstIndex - 1):(firstIndex - 1 + thisNumRecHits)] = inputDataE[(baseInputIndex):(baseInputIndex + thisNumRecHits)] 
      
        firstIndex += thisNumRecHits
      
        # end -- loop over rechits of this photon
  
        progbar.update(1)

    # end -- loop over photons

    progbar.close()

#----------------------------------------------------------------------

def makeOutputDataTracks(indices, inputData, outputData):

    numOutputRows = len(indices)
  
    #----------
    # calculate the total number of output tracks
    #----------
    numOutputTracks = inputData['tracks/numTracks'][indices].sum()
  
    #----------
  
    # now that we know the total number of output rechits, 
    # copy the vectors related to the rechits 
  
    # variables other than the indexing variables 'firstIndex'
    # and 'numTracks'
    otherVarNames = []
  
    for key in inputData.keys():
        if not key.startswith("tracks/"):
            continue

        if key == 'tracks/firstIndex':
            # make sure we have int type here
            outputData[key] = -1 * np.ones(numOutputRows, dtype = 'int32')
        elif key == 'tracks/numTracks':
            # we can just copy this one, filtering by indices
            outputData[key] = inputData[key][indices]
        else:
          # take the input dtype for the output
          outputData[key] = -1 * np.ones(numOutputTracks, dtype = inputData[key].dtype)
    
          otherVarNames.append(key)

    # end -- loop over keys of inputData.tracks

    # note that we keep the Torch/Lua convention of one based indices
    firstIndex = 1
  
    # make local variables for input and output variables
    # to avoid dict/npz file lookups within the loop
    inputDataNumTracks   = inputData['tracks/numTracks']
    inputDataFirstIndex  = inputData['tracks/firstIndex']

    outputDataNumTracks  = outputData['tracks/numTracks']
    outputDataFirstIndex = outputData['tracks/firstIndex']

    #----------
    # calculate tracks/firstIndex
    # and mapping from input to output index ranges for other variables
    #----------

    inputRanges  = [ None ] * numOutputRows
    outputRanges = [ None ] * numOutputRows

    for i in range(numOutputRows):
        index = indices[i]
    
        outputDataFirstIndex[i] = firstIndex

        # note we need to subtract one from both output but
        # also from the input indices 
        inputRanges[i] = slice(inputDataFirstIndex[index] - 1,
                               inputDataFirstIndex[index] - 1 + inputDataNumTracks[index])
                               
        outputRanges[i] = slice(firstIndex - 1,
                                firstIndex - 1 + inputDataNumTracks[index])

        # prepare next iteration
        firstIndex += inputDataNumTracks[index]

    #----------
    # other variables (which need no adjustment like the firstIndex variable)
    #
    # outer loop is on variables so we can avoid repetitive lookups
    # by variable name
    #----------

    for key in inputData.keys():
        if not key.startswith("tracks/"):
            continue

        if key in ('tracks/firstIndex', 'tracks/numTracks'):
            continue
        
        inputVec  = inputData[key]
        outputVec = outputData[key]

        for inputRange, outputRange in zip(inputRanges, outputRanges):
            outputVec[outputRange] = inputVec[inputRange]

        # end of loop over photons
  
    # end of loop over variables

#----------------------------------------------------------------------

def makeOutputData(indices, inputData, checkFirstIndex = True):

    # copy everything except the rechits
    outputData = makeOutputVar(inputData, indices, [ 'X/*', 'tracks/*' ])
  
    makeOutputDataRecHits(indices, inputData, outputData)

    import mergeRecHits
    
    if checkFirstIndex:
        isOk = mergeRecHits.checkFirstIndex(outputData['X/firstIndex'], outputData['X/numRecHits'])
        assert isOk, "internal error with rechits firstIndex"

    if 'tracks/numTracks' in inputData:
        makeOutputDataTracks(indices, inputData, outputData)

        if checkFirstIndex:
            isOk = mergeRecHits.checkFirstIndex(outputData['tracks/firstIndex'], outputData['tracks/numTracks'])
            assert isOk, "internal error with tracks firstIndex"

    return outputData

#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------

ARGV = sys.argv[1:]
assert len(ARGV) == 1

np.random.seed(randSeed)

inputFile = ARGV.pop(0)

#----------
# generate output file test names
#----------
if inputFile.endswith(".npz"):
    outputFilePrefix = inputFile[:-4]
else:
    outputFilePrefix = inputFile

outputFileTrain = outputFilePrefix + "-train.npz"
outputFileTest  = outputFilePrefix + "-test.npz"

#----------

inputData = np.load(inputFile)
numEvents = len(inputData['y'])
print "have", numEvents,"photons"


# throw a random number for each event 
randVals = np.random.rand(numEvents)
trainIndices = np.arange(numEvents)[np.where(randVals > testFraction)]
testIndices = np.arange(numEvents)[np.where(randVals <= testFraction)]

assert len(trainIndices) + len(testIndices) == numEvents

#----------
# create and fill the output tensors
#----------

print "filling train dataset"
outputDataTrain = makeOutputData(trainIndices, inputData)

print "writing train dataset (",len(trainIndices),"photons) to",outputFileTrain

# note that this is uncompressed
np.savez(outputFileTrain, **outputDataTrain)


print "filling test dataset"
outputDataTest = makeOutputData(testIndices, inputData)

print "writing test dataset (", len(testIndices),"photons) to",outputFileTest
np.savez(outputFileTest, **outputDataTest)
