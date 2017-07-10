#!/usr/bin/env python

import time
import numpy as np
import os, sys

from sklearn.metrics import roc_auc_score

import tqdm

from Timer import Timer

from utils import sgdWithLearningRateDecay, iterate_minibatches

sys.path.append(os.path.expanduser("~/torchio")); import torchio

#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------

# parse command line arguments
import argparse

parser = argparse.ArgumentParser(prog='train01.py',
                                 formatter_class = argparse.ArgumentDefaultsHelpFormatter,
                                 )

parser.add_argument('--opt',
                    dest = "optimizer",
                    type = str,
                    choices = [ 'adam', 
                                'sgd',
                                ],
                    default = 'adam',
                    help='optimizer to use (default: %(default)s)'
                    )

parser.add_argument('--print-model-only',
                    dest = "printModelOnlyOutput",
                    type = str,
                    default = None,
                    help='only write the model graphviz file (given after this option) and exit',
                    metavar = 'file.gv',
                    )

parser.add_argument('--dump-inputs-only',
                    dest = "dumpInputsOnlyDir",
                    type = str,
                    default = None,
                    help='only dump expanded dumps to the given directory and exit. ' + 
                         'WARNING: existing files are overwritten witihout notification.',
                    metavar = 'dir',
                    )

parser.add_argument('--dump-input-size',
                    dest = "dumpInputsSize",
                    type = str,
                    default = None,
                    help='used with --dump-inputs-only: specifiy the number of events or fraction ' + 
                    'of events for which to dump the input data',
                    metavar = 'n',
                    )

parser.add_argument('--monitor-gradient',
                    dest = "monitorGradient",
                    default = False,
                    action = 'store_true',
                    help='write out additional information about the gradient to the results directory',
                    )

parser.add_argument('--max-epochs',
                    dest = "maxEpochs",
                    default = None,
                    type = int,
                    help='stop after the given number of epochs',
                    )

parser.add_argument('--output-dir',
                    dest = "outputDir",
                    default = None,
                    help='manually specify the output directory',
                    )

parser.add_argument('--param',
                    dest = "params",
                    default = [],
                    help='additional python to be evaluated after reading model and dataset file. Can be used to change some parameters. Option can be specified multiple times.',
                    action = 'append',
                    )

# TODO: do we also need to be able to specify train and test sizes individually ?
parser.add_argument('--size',
                    dest = "size",
                    default = None,
                    type = float,
                    help='override train and test sample size (e.g. for faster testing): integer value > 1: absolute number of samples to use, float value in the range 0..1: fraction of sample to use',
                    )

parser.add_argument('modelFile',
                    metavar = "modelFile.py",
                    type = str,
                    nargs = 1,
                    help='file with model building code',
                    )

parser.add_argument('dataFile',
                    metavar = "dataFile.py",
                    type = str,
                    nargs = 1,
                    help='file with data loading code',
                    )

options = parser.parse_args()

#----------

if options.printModelOnlyOutput and options.dumpInputsOnlyDir:
    print >> sys.stderr,"--print-model-only and --dump-inputs-only are mutually exclusive"
    sys.exit(1)

if options.dumpInputsSize != None:
    if not options.dumpInputsOnlyDir:
        print >> sys.stderr,"--dump-input-size requires --dump-inputs-only"
        sys.exit(1)

    # try to convert to an integer first
    try:
        options.dumpInputsSize = int(options.dumpInputsSize)
    except ValueError:
        try:
            # now try float
            options.dumpInputsSize = float(options.dumpInputsSize)
        except ValueError:
            print >> sys.stderr,"unrecognized format for --dump-input-size value",options.dumpInputsSize
            sys.exit(1)

#----------

batchsize = 32

# if not None, set average background
# weight to one and the average signal
# weight such that the sum of signal
# weights is sigToBkgFraction times the
# number of background weights
sigToBkgFraction = None

#----------

execfile(options.modelFile[0])
execfile(options.dataFile[0])

for param in options.params:
    # python 2
    exec param

#----------
print "building model"
input_vars, model = makeModel()

# produce network model in graphviz format
import draw_net
dot = draw_net.get_pydot_graph(lasagne.layers.get_all_layers(model), verbose = True)

if options.printModelOnlyOutput != None:
    # just generate the graphviz output and exit
    dot.write(options.printModelOnlyOutput, format = "raw")
    print "wrote model description to", options.printModelOnlyOutput
    sys.exit(0)

#----------
# initialize output directory
#----------

if options.outputDir == None:
    options.outputDir = "results/" + time.strftime("%Y-%m-%d-%H%M%S")

if not os.path.exists(options.outputDir):
    os.makedirs(options.outputDir)

#----------
# try to set the process name
#----------
try:
    import procname
    procname.setprocname("train " + 
                         os.path.basename(options.outputDir.rstrip('/')))
except ImportError, ex:
    pass

#----------
# setup logging
#----------
logfile = open(os.path.join(options.outputDir, "train.log"), "w")

fouts = [ sys.stdout, logfile ]

#----------

print "loading data"

doPtEtaReweighting = globals().get("doPtEtaReweighting", False)

cuda = True

# data e.g. to be shared between train and test dataset
datasetAuxData = {}

with Timer("loading training dataset...") as t:
    trainData, trsize = datasetLoadFunction(dataDesc['train_files'], options.size, 
                                            cuda = cuda, 
                                            isTraining = True,
                                            reweightPtEta = doPtEtaReweighting,
                                            logStreams = fouts,
                                            returnEventIds = False,
                                            auxData = datasetAuxData)
with Timer("loading test dataset...") as t:
    testData,  tesize = datasetLoadFunction(dataDesc['test_files'], options.size, cuda, 
                                            isTraining = False,
                                            reweightPtEta = False,
                                            logStreams = fouts,
                                            returnEventIds = False,
                                            auxData = datasetAuxData
                                            )


#----------
# write training file paths to result directory
#----------

fout = open(os.path.join(options.outputDir, "samples.txt"), "w")
for fname in dataDesc['train_files']:
    print >> fout, fname
fout.close()

#----------

for fout in fouts:
    print >> fout, "doPtEtaReweighting=",doPtEtaReweighting

### print "----------"
### print "model:"
### print model.summary()
### print "----------"
### print "the model has",model.count_params(),"parameters"
### 
### print >> logfile,"----------"
### print >> logfile,"model:"
### model.summary(file = logfile)
### print >> logfile,"----------"
### print >> logfile, "the model has",model.count_params(),"parameters"
### logfile.flush()

#----------
# write graphviz output to results directory
#----------
networkGraphvizFname = os.path.join(options.outputDir, "model")
dot.write(networkGraphvizFname + ".gv",  format = "raw")
dot.write(networkGraphvizFname + ".pdf", format = "pdf")

# runs dot externally (assume graphviz is installed on the machine)
if False:
    for suffix in ("svg",):
        draw_net.draw_to_file(lasagne.layers.get_all_layers(model), 
                              os.path.join(options.outputDir, "model." + suffix))

#----------

target_var = T.imatrix('targets')


# these are of type theano.tensor.var.TensorVariable
train_prediction = lasagne.layers.get_output(model)

#----------
# get the number of output nodes of the model
#----------
outputShape = lasagne.layers.get_output_shape(model)
assert len(outputShape) == 2
assert outputShape[0] == None
numOutputNodes = outputShape[1]

#----------
# check whether we have one or two outputs 
#----------
if numOutputNodes == 1:
    lossFunc = binary_crossentropy

    trainWeights = trainData['weights'].reshape((-1,1))
    testWeights  = testData['weights'].reshape((-1,1))

    weight_var = T.matrix('weights')
elif numOutputNodes == 2:

    # we have two outputs (typically from a softmax output layer)
    lossFunc = categorical_crossentropy

    trainWeights = trainData['weights'].reshape((-1,))
    testWeights  = testData['weights'].reshape((-1,))

    weight_var = T.vector('weights')

else:
    raise Exception("don't know how to handle %d output nodes" % numOutputNodes)

#----------
# pt/eta reweighting
#----------
if doPtEtaReweighting:
    origTrainWeights = trainData['weightsBeforePtEtaReweighting']
else:
    # they're the same
    origTrainWeights = trainWeights

#----------
# fix ratio of sum of signal and background
# events if requested
#----------

if sigToBkgFraction != None:
    
    # normalize average background weight to one
    # (just for the training dataset)
    bkgIndices = trainData['labels'] == 0

    numBkgEvents = sum(bkgIndices)
    sumBkgWeights = float(sum(trainWeights[bkgIndices]))

    trainWeights[bkgIndices] *= numBkgEvents / sumBkgWeights

    #----------
    sigIndices = trainData['labels'] == 1

    sumSigWeights = float(sum(trainWeights[sigIndices]))

    # note that we now multiply with numBkgEvents
    # instead of the original sumBkgWeights
    targetSumSigWeights = sigToBkgFraction * numBkgEvents

    trainWeights[sigIndices] *= targetSumSigWeights / sumSigWeights

    for fout in fouts:
        print >> fout, "sum train sig weights:",sum(trainWeights[sigIndices]),"train sum bkg weights:",sum(trainWeights[bkgIndices])

#----------
# write out BDT/MVA id labels (for performance comparison)
#----------
for name, output in (
    ('train', trainData['mvaid']),
    ('test',  testData['mvaid']),
    ):
    np.savez(os.path.join(options.outputDir, "roc-data-%s-mva.npz" % name),
             # these are the BDT outputs
             output = output,
             )

# save weights (just once, we assume the ordering of the events is always the same)
np.savez(os.path.join(options.outputDir, "weights-labels-train.npz"),
         trainWeight = trainWeights,             
         origTrainWeights = origTrainWeights,
         doPtEtaReweighting = doPtEtaReweighting,
         label = trainData['labels'],
         )
np.savez(os.path.join(options.outputDir, "weights-labels-test.npz"),
         weight = testWeights,             
         label = testData['labels'],
         )


#----------
    
train_loss       = aggregate(lossFunc(train_prediction, target_var), mode = "mean", weights = weight_var)

# deterministic = True is e.g. set to replace dropout layers by a fixed weight
test_prediction = lasagne.layers.get_output(model, deterministic = True)
test_loss       = aggregate(lossFunc(test_prediction, target_var), mode = "mean", weights = weight_var)

# method for updating weights
params = lasagne.layers.get_all_params(model, trainable = True)

train_loss_grad  = theano.grad(train_loss, params)
if globals().has_key('modelParams') and modelParams.has_key('maxGradientNorm'):
    # clip the gradient, see http://lasagne.readthedocs.io/en/latest/modules/updates.html#lasagne.updates.total_norm_constraint
    train_loss_grad = lasagne.updates.total_norm_constraint(train_loss_grad, modelParams['maxGradientNorm'])


#----------
for fout in fouts:
    print >> fout, "using",options.optimizer,"optimizer"

if options.optimizer == 'adam':
    updates = adam(train_loss_grad, params)
elif options.optimizer == 'sgd':

    # parameters taken from Torch examples,
    # should be equivalent
    # but this does not take into account the minibatch size 
    # (i.e. 32 times fewer evaluations of this function, learning
    # rate decays 32 times slower) ?
    updates = sgdWithLearningRateDecay(train_loss_grad, params,
                                       learningRate = 1e-3,
                                       learningRateDecay = 1e-7)

    # original torch parameters:                 
    # optimState = {
    #    -- learning rate at beginning
    #    learningRate = 1e-3,
    #    weightDecay = 0,
    #    momentum = 0,
    #    learningRateDecay = 1e-7
    # }


else:
    raise Exception("internal error")

#----------
# build / compile the goal functions
#----------

with Timer("compiling train dataset loss function...", fouts) as t:
    train_function = theano.function(input_vars + [ target_var, weight_var ], train_loss, updates = updates, name = 'train_function')

    if options.monitorGradient:
        # see e.g. http://stackoverflow.com/a/37384861/288875
        get_train_function_grad = theano.function(input_vars + [ target_var, weight_var ], train_loss_grad)

with Timer("compiling test dataset loss function...", fouts) as t:
    test_function  = theano.function(input_vars + [ target_var, weight_var ], test_loss)

# function to calculate the output of the network
with Timer("compiling network output function...", fouts) as t:
    test_prediction_function = theano.function(input_vars, test_prediction)

#----------
# convert targets to integers (needed for softmax)
#----------

for data in (trainData, testData):
    data['labels'] = data['labels'].astype('int32').reshape((-1,1))

    # check whether we have two outputs 
    if numOutputNodes == 2:
        # we have two outputs (typically from a softmax output layer)
        # we set the second output target values to 1 - labels
        data['labels'] = np.column_stack([ data['labels'], 1 - data['labels'] ])

#----------
# produce test and training input once
#----------
# assuming we have enough memory 
#
# TODO: can we use slicing instead of unpacking these again for the minibatches ?
with Timer("unpacking training dataset...", fouts) as t:
    trainInput = makeInput(trainData, range(len(trainData['labels'])), inputDataIsSparse = True)
    assert len(trainInput) == len(input_vars), "number of sets of values (%d) is not equal to number of input variables (%d)" % (len(trainInput), len(input_vars))

with Timer("unpacking test dataset...", fouts) as t:
    testInput  = makeInput(testData, range(len(testData['labels'])), inputDataIsSparse = True)

train_output = np.zeros(len(trainData['labels']))

#----------
# try to serialize the model structure itself
# will not work if used e.g. on CPU instead of GPU etc.
import cPickle as pickle
pickle.dump(
    dict(model = model,
         input_vars = input_vars), open(os.path.join(options.outputDir,
                                                     "model-structure.pkl"),"w"))
#----------
# dump input data
#----------
if options.dumpInputsOnlyDir:
    for inp, data, weights, label in (
        (trainInput, trainData, trainWeights, 'train'),
        (testInput,  testData,  testWeights, 'test')):

        print "dumping input data for",label,"sample"
        
        # save in pickled format so we can have arbitrary structures
        # (unlike np.savez(..) which in principle could work also)
        foutName = os.path.join(options.dumpInputsOnlyDir,
                                 "input-%s.npz" % label)
        
        outputData = {}

        # limit size of inputs
        if options.dumpInputsSize != None:

            if options.dumpInputsSize < 1:
                # a fraction was specified
                thisSize = int(len(item) * options.dumpInputsSize + 0.5)
            else:
                # an absolute size was specified
                thisSize = min(len(data['labels']), options.dumpInputsSize)

            inp = [ item[:thisSize] for item in inp ]
        else:
            thisSize = len(data['labels'])

        for index,item in enumerate(inp):
            outputData['input/%03d' % index] = item
            
        outputData['labels'] = data['labels'][:thisSize]
        outputData['mvaid']  = data['mvaid'][:thisSize]
        
        # these are the weights actually used for training
        # (i.e. after pt/eta reweighting)
        outputData['weights'] = weights[:thisSize]

        np.savez(foutName, **outputData)
        print "wrote",foutName

    sys.exit(0)

#----------

print "params=",params
print
print 'starting training at', time.asctime()


epoch = 1
while True:

    #----------

    if options.maxEpochs != None and epoch > options.maxEpochs:
        break

    #----------

    nowStr = time.strftime("%Y-%m-%d %H:%M:%S")
        
    for fout in fouts:

        print >> fout, "----------------------------------------"
        print >> fout, "starting epoch %d at" % epoch, nowStr
        print >> fout, "----------------------------------------"
        print >> fout, "output directory is", options.outputDir
        fout.flush()

    #----------
    # check if we should only train on a subset of indices
    #----------
    if globals().has_key("adaptiveTrainingSample") and adaptiveTrainingSample:
        assert globals().has_key('trainEventSelectionFunction'), "function trainEventSelectionFunction(..) not defined"

        if epoch == 1:
            for fout in fouts:
                print >> fout, "using adaptive training event selection"

        selectedIndices = trainEventSelectionFunction(epoch, 
                                                      trainData['labels'],
                                                      trainWeights,
                                                      train_output,
                                                      )

        # make sure this is an np.array(..)
        selectedIndices = np.array(selectedIndices)
    else:
        selectedIndices = np.arange(len(trainData['labels']))

    #----------
    # training 
    #----------

    sum_train_loss = 0
    train_batches = 0

    if len(selectedIndices) < len(trainData['labels']):
        for fout in fouts:
            print >> fout, "training on",len(selectedIndices),"out of",len(trainData['labels']),"samples"

    progbar = tqdm.tqdm(total = len(selectedIndices), mininterval = 1.0, unit = 'samples')

    # magnitude of overall gradient. index is minibatch within epoch
    if options.monitorGradient:
        gradientMagnitudes = []

    startTime = time.time()
    for indices, targets in iterate_minibatches(trainData['labels'], batchsize, shuffle = True, selectedIndices = selectedIndices):

        # inputs = makeInput(trainData, indices, inputDataIsSparse = True)

        inputs = [ inp[indices] for inp in trainInput]

        thisWeights = trainWeights[indices]

        # this also updates the weights ?
        sum_train_loss += train_function(* (inputs + [ targets, thisWeights ]))

        # this leads to an error
        # print train_prediction.eval()

        if options.monitorGradient:
            # this actually returns a list of CudaNdarray objects
            gradients = get_train_function_grad(* (inputs + [ targets, thisWeights ]))

            gradients = [ np.ndarray.flatten(np.asarray(grad)) for grad in gradients ]

            # produce the overall gradient
            gradient = np.concatenate(gradients)
            gradientMagnitudes.append(np.linalg.norm(gradient))

        train_batches += 1

        progbar.update(batchsize)

    # end of loop over minibatches
    progbar.close()

    #----------

    deltaT = time.time() - startTime

    for fout in fouts:
        print >> fout
        print >> fout, "time to learn 1 sample: %.3f ms" % ( deltaT / len(trainWeights) * 1000.0)
        print >> fout, "time to train entire batch: %.2f min" % (deltaT / 60.0)
        print >> fout
        print >> fout, "avg train loss:",sum_train_loss / float(len(selectedIndices))
        print >> fout
        fout.flush()

    #----------
    # save gradient magnitudes
    #----------

    if options.monitorGradient:
        np.savez(os.path.join(options.outputDir, "gradient-magnitudes-%04d.npz" % epoch),
                 gradientMagnitudes = np.array(gradientMagnitudes),
                 )

    #----------
    # calculate outputs of train and test samples
    #----------

    evalBatchSize = 10000

    outputs = []

    for input in (trainInput, testInput):
        numSamples = input[0].shape[0]
        
        thisOutput = np.zeros(numSamples)

        for start in range(0,numSamples,evalBatchSize):
            end = min(start + evalBatchSize,numSamples)

            thisOutput[start:end] = test_prediction_function(
                *[ inp[start:end] for inp in input]
                )[:,0]

        outputs.append(thisOutput)

    train_output, test_output = outputs
            
    # evaluating all at once exceeds the GPU memory in some cases
    # train_output = test_prediction_function(*trainInput)[:,1]
    # test_output = test_prediction_function(*testInput)[:,1]

    #----------
    # calculate AUCs
    #----------

    for name, predictions, labels, weights in  (
        # we use the original weights (before pt/eta reweighting)
        # here for printing for the train set, i.e. not necessarily
        # the weights used for training
        ('train', train_output, trainData['labels'], origTrainWeights),
        ('test',  test_output,  testData['labels'],  testWeights),
        ):

        if numOutputNodes == 2:
            # make sure we only have one column
            labels = labels[:,0]

        auc = roc_auc_score(labels,
                            predictions,
                            sample_weight = weights,
                            average = None,
                            )

        for fout in fouts:
            print >> fout
            print >> fout, "%s AUC: %f" % (name, auc)
            fout.flush()

        # write out online calculated auc to the result directory
        fout = open(os.path.join(options.outputDir, "auc-%s-%04d.txt" % (name, epoch)), "w")
        print >> fout, auc
        fout.close()

        # write network output
        np.savez(os.path.join(options.outputDir, "roc-data-%s-%04d.npz" % (name, epoch)),
                 output = predictions,
                 )


    #----------
    # saving the model weights
    #----------

    np.savez(os.path.join(options.outputDir, 'model-%04d.npz' % epoch), 
             *lasagne.layers.get_all_param_values(model))

    #----------
    # prepare next iteration
    #----------
    epoch += 1
