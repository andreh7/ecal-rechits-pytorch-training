#!/usr/bin/env python

from Marker import Marker

#----------------------------------------------------------------------

class PrintHook:
    """ class for debugging the flow of calls to forward().
    Uses indentation to visualize hierarchy of class.
    """

    def __init__(self):
        self.indent = ""

    def inputRepr(self, input):
        if isinstance(input, list) or isinstance(input, tuple):
            # it's a list or tuple
            return [ self.inputRepr(x) for x in input ]
        else:
            return input.size()


    def before(self, module, input):
        # will be called before calling forward()

        # print information about the module and the input sizes
        import sys
        print >> sys.stderr,"%srunning forward on module" % self.indent,

        if isinstance(module, Marker):
            print >> sys.stderr,"marker %s" % module.markerText,
        else:
            # generic module
            print >> sys.stderr,"of type",module.__class__,

        print >> sys.stderr,"input shape:",self.inputRepr(input)

        self.indent += "  "

    def after(self, module, input, output):
        # will be called after calling forward()
        self.indent = self.indent[:-2]


    def registerWith(self, model):
        # register this print hook with the model
        model.apply(lambda module: module.register_forward_pre_hook(self.before))    
        model.apply(lambda module: module.register_forward_hook(self.after))    

#----------------------------------------------------------------------
