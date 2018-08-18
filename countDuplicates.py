#!/usr/bin/env python

# tool to dump a list of sample/run/ls/event numbers
# and count duplicates across samples and within a sample
# to compare samples to others

import sys
import numpy as np
import os

#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------
# parse command line arguments
import argparse

parser = argparse.ArgumentParser(prog='train01.py',
                                 formatter_class = argparse.ArgumentDefaultsHelpFormatter,
                                 )

parser.add_argument('--csv',
                    dest = "csv_output_file",
                    type = str,
                    default = None,
                    metavar = 'list.csv',
                    help='csv file to write event lists to'
                    )

parser.add_argument('inputfile',
                    metavar = "file1.npz",
                    type = str,
                    nargs = "+",
                    help='input files with sparse rechits',
                    )

options = parser.parse_args()

# maps from (run, ls, event) to list of samples
# note that the same event can appear more than once
# in one sample because we can look at more than one
# photon per event

event_to_sample = {}

if options.csv_output_file:
    import csv
    csv_fout = open(options.csv_output_file, "w")
    fieldnames = ["sample","label", "run","ls", "event", "eta","pt"]
    csv_writer = csv.DictWriter(csv_fout, fieldnames = fieldnames)

    # write csv header
    csv_writer.writeheader()

else:
    csv_writer = None

all_sample_events = {}

import collections

for file_index, fname in enumerate(options.inputfile):

    print >> sys.stderr,"opening %s (%d/%d)" % (fname, file_index + 1, len(options.inputfile))

    sample = os.path.basename(fname).split('_')[0]

    data = np.load(fname)

    # to print an event only once
    this_sample_events = all_sample_events.setdefault(sample,set())

    # for counting how many events have more than one entry in the same
    # class (signal/background)
    this_sample_sig_events = collections.Counter()
    this_sample_bg_events = collections.Counter()

    for run, ls, event, label, eta, pt in zip(data['run'], data['ls'], data['event'], data['y'].astype('i4'),
                                     # for debugging
                                     data['phoIdInput/scEta'], data['phoVars/phoEt'],
                                              ):

        the_id = (run, ls, event)

        if label == 0:
            this_sample_bg_events[the_id] += 1
        else:
            this_sample_sig_events[the_id] += 1

        # note that for some events we have e.g. two signal photons etc.
        ## if the_id in this_sample_events:
        ##     continue

        this_sample_events.add(the_id)

        if not csv_writer is None:
            csv_writer.writerow(dict(
                    sample = sample,
                    label = label,
                    run = run, 
                    ls = ls,
                    event = event,
                    eta = eta,
                    pt = pt))
                
        if the_id not in event_to_sample:
            event_to_sample[the_id] = set([sample])
        else:
            event_to_sample[the_id].add(sample)

    for name, counts in [
        ('signal', this_sample_sig_events),
        ('background', this_sample_bg_events)]:

        num_non_single_events = len([      
                1 for event_id, count in counts.items() if count >= 2 ])

        total_num_events = sum(counts.values())

        print >> sys.stderr, "events in %s with more than one photon for %s: %d out of %d (%.1f%%)" % (fname, name, num_non_single_events,
                                                                                                     total_num_events,
                                                                                                     num_non_single_events / float(total_num_events) * 100.)


# count events appearing in more than one sample


# maps from frequency to number of events which
# occur in that many different samples
frequency_to_num_events = collections.Counter()
for dataset in event_to_sample.values():
    frequency_to_num_events[len(dataset)] += 1


for key, value in frequency_to_num_events.items():
    print "number of events appearing in %d samples: %d" % (key, value)
    


