#!/usr/bin/env python

# reformats rechits
# 
# we change the format for the rechits here: instead of writing 
# out a table of tables, we write out:
#     
#   - a flat tensor with ALL x, y and energy values (three tensors)
#   - an additional tensor mapping from the photon index
#     to the first index in the above tensors
#   - an additional tensor mapping from the photon index
#     to the number of rechits in this photon (as a convenience)

import numpy as np
import os, re

#----------------------------------------------------------------------

def addSparseRecHits(allData, thisData):
    # no need to convert tensor types with npy (these are already correct)
    #
    # TODO: it may be faster to concatenate all files in once

    xkeys = [ key for key in allData.keys() if key.startswith("X/") ]

    if not xkeys:
  
        # this is the first time we add rechits to allData, just
        # copy the vectors over
        #
        # note that we need to copy the values as 
        # we probably can't add to opened .npz file data
        for key in thisData.keys():
            if key.startswith("X/"):
                allData[key] = [ thisData[key] ]
    else:
        # append the values for x, y, energy and numRecHits 
        # assert allData['X/numRecHits'].sum() == len(allData['X/energy'])
        # assert len(allData['X/firstIndex']) == len(allData['X/numRecHits'])
        # assert allData['X/firstIndex'][-1] + allData['X/numRecHits'][-1] - 1 == len(allData['X/energy'])

        # we must add an offset to firstIndex
        numPhotonsBefore     = sum([ len(item) for item in allData['X/firstIndex'] ])
        numRecHitsBefore     = sum([ len(item) for item in allData['X/energy'] ] )
  
        thisNumPhotons       = len(thisData['X/firstIndex'])
        thisNumRecHits       = len(thisData['X/energy'])
  
        assert thisData['X/firstIndex'][-1] + thisData['X/numRecHits'][-1] - 1 == thisNumRecHits

        allData['X/x'].append(thisData['X/x'])
        allData['X/y'].append(thisData['X/y'])
        allData['X/energy'].append(thisData['X/energy'])
        allData['X/numRecHits'].append(thisData['X/numRecHits'])

        # expand the firstIndex field
        allData['X/firstIndex'].append(thisData['X/firstIndex'] + numRecHitsBefore)

        # assert len(allData['X/firstIndex']) == len(allData['X/numRecHits'])

        assert thisData['X/firstIndex'][-1] + thisData['X/numRecHits'][-1] - 1 == len(thisData['X/energy'])

        assert np.all(thisData['X/numRecHits'] >= 1)
  
        # end of loop over photons
  
    # end -- if first time

#----------------------------------------------------------------------

def catItem(item1, item2):
    # note that in the numpy version we do not have nested
    # dicts (in the torch version we had nested tables)

    return np.concatenate([item1, item2])

#----------------------------------------------------------------------

def checkFirstIndex(firstIndexArray, numEntriesArray):
    # checks that firstIndex is the cumulative of the numEntriesArray
    
    # assume there is at least one entry

    fromNumEntries = np.cumsum(np.concatenate([ np.array([1]),
                                                numEntriesArray ]))[:-1]

    result = np.all(firstIndexArray == fromNumEntries)

    if not result:

        diffLocs = np.where(firstIndexArray != fromNumEntries)
        print "diffLocs=",diffLocs
    
        print "first:",firstIndexArray[diffLocs[0]], fromNumEntries[diffLocs[0]]

    return result

#----------------------------------------------------------------------

def addTracks(allData, thisData):
    # convert some tensors (to avoid roundoff errors with indices)
  
    tracksKeys = [ key for key in allData.keys() if key.startswith("tracks/") ]

    if not tracksKeys:
  
        # this is the first time we add rechits to allData, just
        # copy the vectors over
  
        for key in thisData.keys():
            if key.startswith("tracks/"):
                allData[key] = [ thisData[key] ]
    else:

        if allData.has_key('tracks/relpt'):
            trackPtName = "tracks/relpt"
        elif allData.has_key('tracks/pt'):
            trackPtName = 'tracks/pt'
        else:
            raise Exception("dont know the name of the tracks pt variable")

        # assert allData['tracks/numTracks'].sum() == len(allData[trackPtName])
        # assert allData['tracks/firstIndex'][-1] + allData['tracks/numTracks'][-1] - 1 == len(allData[trackPtName])

        # append the values for relpt, charge etc.
        # we must add an offset to firstIndex
        numPhotonsBefore     = sum([ len(item) for item in allData['tracks/firstIndex'] ])
        numTracksBefore      = sum([ len(item) for item in allData[trackPtName] ])
  
        thisNumPhotons       = len(thisData['tracks/firstIndex'])
        thisNumRecHits       = len(thisData[trackPtName])
  
        assert thisData['tracks/firstIndex'][thisNumPhotons - 1] + thisData['tracks/numTracks'][thisNumPhotons - 1] - 1 == thisNumRecHits
  
        # concatenate data fields
  
        for varname in thisData.keys():
            if not varname.startswith('tracks/'):
                continue

            if varname != 'tracks/numTracks' and varname != 'tracks/firstIndex':
                allData[varname].append(thisData[varname])

        # end of loop over variables

        allData['tracks/numTracks'].append(thisData['tracks/numTracks'])

        thisDataTracksFirstIndex = thisData['tracks/firstIndex']
        thisDataTracksNumTracks  = thisData['tracks/numTracks']

        #----------
        # expand the firstIndex field
        #----------
        allData['tracks/firstIndex'].append(thisDataTracksFirstIndex + numTracksBefore)

        # for sanity checks
        expectedFirstIndex = 1
  
        assert thisData['tracks/firstIndex'][thisNumPhotons - 1] + thisData['tracks/numTracks'][thisNumPhotons - 1] - 1 == len(thisData[trackPtName])

        for i in range(thisNumPhotons):
            # sanity check of input data
            assert thisDataTracksFirstIndex[i] == expectedFirstIndex
    
            # note that we may have photons without any track nearby
            # (this is NOT the case for rechits on the other hand)
            assert thisDataTracksNumTracks[i] >= 0
    
            if i < thisNumPhotons - 1:
                assert thisDataTracksFirstIndex[i] + thisDataTracksNumTracks[i] == thisDataTracksFirstIndex[i+1]
            else:
                assert thisDataTracksFirstIndex[i] + thisDataTracksNumTracks[i] - 1 == len(thisData[trackPtName]),  \
                 str(thisDataTracksFirstIndex[i] + thisDataTracksNumTracks[i] - 1)  + " " + str(thisData[trackPtName])
    
            # add original firstIndex field
            # allData['tracks/firstIndex'][numPhotonsBefore + i] = thisDataTracksFirstIndex[i] + numTracksBefore

            expectedFirstIndex = expectedFirstIndex + thisDataTracksNumTracks[i]
    
        # end -- loop over photons
  
    # end -- if first time


#----------------------------------------------------------------------

def makeSampleId(fname, numEntries):
    # @return a number determined from the file name 
    fname = os.path.basename(fname)

    fname = fname.split('_')[0]

    num = "".join([ ch for ch in fname if ch >= '0' and ch <= '9' ])

    if num == "":
        num = -1
    else:
        num = int(num, 10)

    return np.ones(numEntries, dtype = 'i4') * num


#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------

if __name__ == '__main__':

    from optparse import OptionParser
    parser = OptionParser("""

      usage: %prog [options] data-directory

              merges npz output files from TorchDumper from multiple jobs into single files

        """
        )

    parser.add_option('--no-tracks',
                        dest = "mergeTracks",
                        action = 'store_false',
                        default = True,
                        help='do not merge track information'
                        )

    parser.add_option('--check-first-index',
                      dest = "checkFirstIndex",
                      default = False,
                      action = 'store_true',
                      help='checks the merging of the firstIndex arrays'
                        )

    (options, ARGV) = parser.parse_args()

    if len(ARGV) != 1:
        print >> sys.stderr, "must specify exactly one data directory to work on"
        sys.exit(1)

    dataDir = ARGV.pop(0)

    #----------

    for subdet in ("barrel", "endcap"):

        #----------
        # find all .npz files matching a given pattern and group them
        #----------

        # maps from basename to list of files
        fileGroups = {}

        # example name: output4/GJet40toInf_rechits-endcap_96.t7
        import glob
        inputFnames = glob.glob(dataDir + "/*-" + subdet + '_*.npz')

        if not inputFnames:
            print "no input files found for " + subdet + " in " + dataDir
            continue

        for fname in inputFnames:

            mo = re.search("/(.*)_rechits-" + subdet + "_(\d+)\.npz$", fname)
            assert mo, "unexpected filename format " + fname

            baseName = mo.group(1)
            number = int(mo.group(2))

            fileGroups.setdefault(baseName, {})

            assert(not fileGroups[baseName].has_key(number))
            fileGroups[baseName][number] = fname

        # end of loop over input file names

        #----------

        for baseName, fileNames in fileGroups.items():

            outputFname = os.path.join(dataDir, baseName + "_rechits-" + subdet + ".npz")
            print "merging files into",outputFname

            #----------
            # traverse the list increasing file index
            #----------
            allData = None

            for fileIndex in sorted(fileNames.keys()):

                fname = fileNames[fileIndex]

                print "opening",fname, fileIndex,"/",len(fileNames),
                thisData = np.load(fname)

                numPhotons = len(thisData['y'])

                print numPhotons,"photons"

                # skip empty files, at the moment they make some of the checks
                # above fail
                if numPhotons == 0:
                    continue

                recHitsAdded = False
                tracksAdded  = False

                if allData == None:

                    allData = {}

                    allData['sample'] = [ makeSampleId(fname, len(thisData['y'])) ]

                    for key in thisData.keys():

                        value = thisData[key]

                        if key != 'genDR':
                            if key.startswith('X/'):
                                if not recHitsAdded:
                                    # we only support sparse format here
                                    addSparseRecHits(allData, thisData)
                                    recHitsAdded = True

                            elif key.startswith('tracks/'): 
                                if not tracksAdded and options.mergeTracks:
                                    addTracks(allData, thisData)
                                    tracksAdded = True
                            else:
                                # just copy the data
                                allData[key] = [ value ]

                        # if not genDR

                    # end loop over all items in the dict

                else:
                    # append to existing data

                    allData['sample'].append(makeSampleId(fname, len(thisData['y'])))

                    for key in thisData.keys():
                        value = thisData[key]

                        if key != 'genDR':
                            if key.startswith('X/'):
                                if not recHitsAdded:
                                    # we only support sparse format here
                                    addSparseRecHits(allData, thisData)
                                    recHitsAdded = True
                            elif key.startswith('tracks/'):
                                if not tracksAdded and options.mergeTracks:
                                    addTracks(allData, thisData)
                                    tracksAdded = True
                            else:
                                # normal concatenation
                                allData[key].append(thisData[key])

                        # end if not genDR

                    # end loop over all items in the dict
                # end if not first

            # end of loop over file names for this base name

            #----------
            # perform np.concatenate only at the end
            #----------

            for index, (key, value) in enumerate(allData.items()):
                print "concatenating %d/%d" % (index + 1, len(allData))

                assert isinstance(value,list),"item " + key + " is not of type list but " + str(value.__class__)

                allData[key] = np.concatenate(value)

            #----------
            # check firstIndex values after merging if requested
            #----------
            if options.checkFirstIndex:
                if allData.has_key('X/firstIndex'):
                    recHitsOk = checkFirstIndex(allData['X/firstIndex'], allData['X/numRecHits'])
                    print "recHitsOk:",recHitsOk
                else:
                    recHitsOk = True

                if allData.has_key('tracks/firstIndex'):
                    tracksOk = checkFirstIndex(allData['tracks/firstIndex'], allData['tracks/numTracks'])
                    print "tracksOk:",tracksOk
                else:
                    tracksOk = True

                if not (recHitsOk and tracksOk):
                    print >> sys.stderr,"at least one firstIndex varible is incorrect, exiting"
                    sys.exit(1)

            #----------
            # write out
            #----------
            print "writing",outputFname,"(",len(allData['y']),"photons )"

            # it looks like **kwds is able to deal with slashes in the key names of the dict...
            np.savez(outputFname, **allData)

        # end of looping over base names (processes)

    # end of loop over barrel/endcap

