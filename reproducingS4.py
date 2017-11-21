#!/usr/bin/env python

# test trying to recalculate S4 from rechits input
#
# see also ../lasagne-training/reproducingTrackIso.py

datasetDir = '../data/2017-03-19-npy-mass-cut'

#----------------------------------------------------------------------

import numpy as np
import glob

np.set_printoptions(linewidth=300, suppress = True, precision = 2)

#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------

# load and combine files: note that the 'firstIndices'
# array must be modified

data = {}

fnames = glob.glob(datasetDir + "/*-barrel*.npz")

# data['fileIndex'] = []
# data['fileOffsetPhotons'] = []

rechits = []
s4values = []

rechits_dim = (7,23)

from rechitmodelutils import RecHitsUnpacker
unpacker = RecHitsUnpacker(
    rechits_dim[0], # width
    rechits_dim[1], # height
    
    # for shifting 18,18 to 4,12
    recHitsXoffset = -18 + rechits_dim[0] / 2 + 1,
    recHitsYoffset = -18 + rechits_dim[1] / 2 + 1,
    )


# DEBUG
# fnames = fnames[:1]
fnames = ['../data/2017-03-19-npy-mass-cut/GJet40toInf_rechits-barrel-train.npz']

for fileIndex, fname in enumerate(fnames):

    print "opening %s file %d/%d" % (fname, fileIndex + 1, len(fnames))
    
    thisData = np.load(fname)

    wrapper = dict(
        rechits = dict(
            firstIndex = thisData['X/firstIndex'],
            numRecHits = thisData['X/numRecHits'],
            x          = thisData['X/x'],
            y          = thisData['X/y'],
            energy     = thisData['X/energy'],
            )
        )

    rechits.append(unpacker.unpack(wrapper, range(len(wrapper['rechits']['firstIndex']))))

    s4values.append(thisData['phoIdInput/s4'])

# end of loop over files

s4values = np.concatenate(s4values)
rechits = np.concatenate(rechits)

# remove unit dimension
rechits = rechits.squeeze(1)        

#----------

conv_weights = np.ones((2,2))

recalculated_s4values = []

import scipy.signal

print "calculating s4 values"
for i in range(len(rechits)):
    # original thought: max (2,2) sum over sum of rechits
    ### recalculated_s4values.append(
    ###     scipy.signal.convolve2d(rechits[i], conv_weights).max() / rechits[i].sum()
    ###     )

    # update: same but only in a 5x5 window (for both the numerator
    # and the denominator)

    # assume 7x23 window -> center is at (3,11)
    # -> (5x5) center is (1..5) x (9..13)
    tower = rechits[i][1:6,9:14]

    recalculated_s4values.append(
        scipy.signal.convolve2d(tower, conv_weights).max() / tower.sum()
        )

recalculated_s4values = np.array(recalculated_s4values)


ratios = recalculated_s4values / s4values

print "plotting"
import matplotlib.pyplot as plt
plt.figure()
plt.hist(ratios, bins = 100)
plt.grid()

# zoom
plt.figure()
plt.hist(ratios[np.fabs(ratios-1) < 0.1], bins = 100); 
plt.grid()

margin = 0.01

fraction_within_margin = (np.abs(ratios - 1) < margin).sum() / float(len(ratios))

plt.title("fraction of values within +/-%.1f%%: %.1f%%" % (margin * 100,
          fraction_within_margin * 100))

plt.show()



 
