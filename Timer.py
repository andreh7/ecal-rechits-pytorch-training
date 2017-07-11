#!/usr/bin/env python

import sys, time

#----------------------------------------------------------------------

# see http://preshing.com/20110924/timing-your-code-using-pythons-with-statement/
class Timer:    
    # timer printing a message

    def __init__(self, msg = None, fout = None):
        self.msg = msg

        if fout == None:
            fout = sys.stdout

        if isinstance(fout, list):
            # TODO: should make this for any iterable
            self.fouts = list(fout)
        else:
            self.fouts = [ fout ]


    def __enter__(self):
        if self.msg != None:
            for fout in self.fouts:
                print >> fout, self.msg,
                fout.flush()

        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start
        if self.msg != None:
            for fout in self.fouts:
                print >> fout, "%.1f seconds" % self.interval
#----------------------------------------------------------------------
