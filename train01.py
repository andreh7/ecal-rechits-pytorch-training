#!/usr/bin/env python

import time
import numpy as np
import os, sys, re

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from sklearn.metrics import roc_auc_score

import tqdm

from Timer import Timer

from torch.utils.data import DataLoader

import cProfile

#----------------------------------------------------------------------

import signal

stopFlag = False

def breakHandler(signal, frame):
    # handler for stopping gracefully when CTRL-C was pressed
    global stopFlag
    stopFlag = True
    print >> sys.stderr,"CTRL-C pressed, exiting soon..."

    # do we need to reregister ?


#----------------------------------------------------------------------

def loadCheckpoint(outputDir, model, optimizer):
    # loads model weights and optimizer state from
    # the latest checkpoint file found in the output directory
    # 

    # find the highest numbered checkpoint file
    import glob
    latestEpoch = None
    latestEpochFile = None
    for fname in glob.glob(os.path.join(outputDir, "checkpoint-*.torch")):

        mo = re.match("checkpoint-(\d+)\.torch$", os.path.basename(fname))

        if mo is None:
            continue

        epoch = int(mo.group(1))

        if latestEpoch is None or epoch > latestEpoch:
            latestEpoch = epoch
            latestEpochFile = fname

    if latestEpoch is None:
        raise Exception("could not find any checkpoint file in directory %s" % outputDir)

    #----------
    # load the states from the checkpoint file
    #----------
    # (see also https://github.com/pytorch/examples/blob/7d0d413425e2ee64fcd0e0de1b11c5cca1f79f4d/imagenet/main.py#L98 )
    
    state = torch.load(latestEpochFile)

    if options.cuda:
        optimizer.cuda(options.cudaDevice)
        model.cuda(options.cudaDevice)
    else:
        optimizer.cpu()
        model.cpu()
    
    return latestEpoch


#----------------------------------------------------------------------

def unpackLoadedBatch(tensors, cuda, volatile):
    # unpacks a batch of weights, targets and input variables
    # according to our ordering convention and
    # turns them into variables
    #
    # @param volatile is typically set to True for inference
    weights      = tensors[0]
    targets      = tensors[1]
    inputTensors = tensors[2:]

    # looks like we have to convert the tensors to CUDA ourselves ?
    if options.cuda:
        inputVars = [ Variable(x.cuda(options.cudaDevice), requires_grad = False, volatile = volatile) for x in inputTensors ]

        # TODO: should we also use volatile here ?
        targetVar = Variable(targets).cuda(options.cudaDevice)
    else:
        inputVars = [ Variable(x, requires_grad = False) for x in inputTensors ]
        targetVar = Variable(targets)

    return weights, targetVar, inputVars

#----------------------------------------------------------------------

def epochIteration():
    # iterates once over the training sample
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

    # put model in training mode
    model.train()

    if len(selectedIndices) < len(trainData['labels']):
        for fout in fouts:
            print >> fout, "training on",len(selectedIndices),"out of",len(trainData['labels']),"samples"

    progbar = tqdm.tqdm(total = len(selectedIndices), mininterval = 1.0, unit = 'samples')

    # magnitude of overall gradient. index is minibatch within epoch

    startTime = time.time()

    # see http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    for batchIndex, tensors in enumerate(dataloader):

        if stopFlag:
            # do not save the current state of the model, the previous
            # iteration should be good enough
            return

        # batchIndex is an int
        #
        # batchDatas is typically a list of torch tensors whose
        # first dimension has the size of the minibatch

        weights, targetVar, inputVars = unpackLoadedBatch(tensors, options.cuda, volatile = False)

        # skip the last batch which may be odd-sized, in particular
        # does not fit the size of the weights tensor used for the loss
        if weights.size(0) != options.batchsize:
            continue

        if optimizer != None:
            optimizer.zero_grad()

        # forward through the network
        # at this point trainInput is still a numpy array
        output = model(inputVars)

        # update weights for loss function
        # note that we use the argument size_average = False so the
        # weights must average to one
        # (currently we average the weights over minibatch
        # as we most likely effectively did in the Lasagne training
        # but eventually we should average them over the entire training sample..)
        weightsTensor[:] = weights / weights.sum()

        # calculate loss
        if lossFunc == None:
            # just calculate the AUC of the training and test sample
            break

        loss = lossFunc.forward(output, targetVar)

        sum_train_loss += loss.data[0]

        # backpropagate and update weights
        loss.backward()

        # update learning rate
        optimizer.step()

        train_batches += 1

        progbar.update(options.batchsize)

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
    # calculate outputs of train and test samples
    # (note that we also recalculate the outputs
    # on the train samples to have them calculated
    # with a consistent set of network weights)
    #----------

    # put model into evaluation mode
    model.eval()

    evalBatchSize = 10000

    outputs = []

    save_targets_now = epoch == 1 and save_targets

    #----------
    def make_output_tensor(num_samples, output_example_shape):
        output_example_shape = list(output_example_shape)
        return np.zeros([num_samples] + output_example_shape[1:])
        
    #----------

    for dataset_name, dataset in (
        ("train", trainDataSet), 
        ("test", testDataSet)):

        evalDataLoader = DataLoader(dataset, batch_size = evalBatchSize, shuffle = False)

        numSamples = len(dataset)
        thisOutput = None

        if save_targets_now:
            # assume scalar target variable
            targets = None

        for batchIndex, tensors in enumerate(evalDataLoader):
            start = batchIndex * evalBatchSize
            end = min(start + evalBatchSize,numSamples)

            weights, targetVar, inputVars = unpackLoadedBatch(tensors, options.cuda, volatile = True)

            # forward pass
            output = model(inputVars)

            if options.cuda:
                output = output.cpu()

            #----------
            # make output tensor now that we know the shape
            #----------
            if thisOutput is None:
                thisOutput = make_output_tensor(numSamples, output.size())

            thisOutput[start:end] = output.data.numpy()

            #----------

            if save_targets_now:
                if options.cuda:
                    targetVar = targetVar.cpu()

                if targets is None:
                    # this is the first time, create targets with correct shape
                    targets = make_output_tensor(numSamples, targetVar.size())

                targets[start:end] = targetVar.data.numpy()

        outputs.append(thisOutput)
        
        #----------
        # write out actual target values for regression problems
        #----------
        if save_targets_now:
            print "WRITING",os.path.join(options.outputDir, "targets-" + dataset_name + ".npz")
            np.savez(os.path.join(options.outputDir, "targets-" + dataset_name + ".npz"),
                     target = targets)


    train_output, test_output = outputs
            
    # evaluating all at once exceeds the GPU memory in some cases
    # train_output = test_prediction_function(*trainInput)[:,1]
    # test_output = test_prediction_function(*testInput)[:,1]

    #----------
    # calculate AUCs
    #----------

    # check if we have an 1D (or 1D compatible) output or not
    # inspired by sklearn.utils.validation.column_or_1d()
    output_shape = np.shape(train_output)
    calculate_auc = len(output_shape) == 1 or (len(output_shape) == 2 and output_shape[1] == 1)
    
    for name, predictions, labels, weights in  (
        # we use the original weights (before pt/eta reweighting)
        # here for printing for the train set, i.e. not necessarily
        # the weights used for training
        ('train', train_output, trainData['labels'], origTrainWeights),
        ('test',  test_output,  testData['labels'],  testWeights),
        ):

        if calculate_auc:

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

    # np.savez(os.path.join(options.outputDir, 'model-%04d.npz' % epoch),
    #          *lasagne.layers.get_all_param_values(model))
    

#----------------------------------------------------------------------

def dumpModelOnnx(model, outputFname, cuda, dataloader):
    import torch.onnx

    # for onnx we need to give some input data
    # need to wrap dataloader in iter(..) because it does not have
    # a next() method (__iter__() alone is not sufficient, 
    # see https://stackoverflow.com/a/33956803/288875 )
    tensors = next(iter(dataloader))

    weights, targetVar, inputVars = unpackLoadedBatch(tensors, cuda, volatile = True)

    torch.onnx.export(model,
                      args = inputVars,
                      f = outputFname,
                      # also write weights out so that
                      # we get the initializer fields
                      # and can distinguish between
                      # variable inputs and trained weights.
                      #
                      # (even if the model is not trained
                      # at the beginning)
                      export_params = True,
                      )

#----------------------------------------------------------------------

def makeDefaultLoss(numOutputNodes, weightsTensor, trainWeights, testWeights):
    # returns the loss function (depending on the number
    # of output nodes) and reshaped train and test weights

    #----------
    # check whether we have one or two outputs 
    #----------
    if numOutputNodes == 1:
        lossFunc = nn.BCELoss(weightsTensor, size_average = False)

        trainWeights = trainWeights.reshape((-1,1))
        testWeights  = testWeights.reshape((-1,1))

    elif numOutputNodes == 2:

        # we have two outputs (typically from a softmax output layer)
        lossFunc = nn.CrossEntropyLoss(weightsTensor, size_average = False)

        trainWeights = trainData['weights'].reshape((-1,))
        testWeights  = testData['weights'].reshape((-1,))

    elif numOutputNodes == None:
        # do not apply a loss function (e.g. useful
        # to calculate the AUC of input variables)
        lossFunc = None

        trainWeights = trainData['weights'].reshape((-1,1))
        testWeights  = testData['weights'].reshape((-1,1))

    else:
        raise Exception("don't know how to handle %d output nodes" % numOutputNodes)

    return lossFunc, trainWeights, testWeights

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


parser.add_argument('--nocuda',
                    # note the inverted logic
                    dest = "cuda",
                    default = True,
                    action = "store_false",
                    help='run on CPU instead of GPU',
                    )


parser.add_argument('--gpu',
                    dest = "cudaDevice",
                    default = 0,
                    type = int,
                    help='index of GPU to run on',
                    )

parser.add_argument('--pprof',
                    dest = "pythonProfiling",
                    default = False,
                    action = 'store_true',
                    help='enable python profiling during training/evaluation (but not initial data loading)',
                    )

parser.add_argument('--resume',
                    default = False,
                    action = 'store_true',
                    help='resume training from latest checkpoint',
                    )

parser.add_argument('--batch-size',
                    default = 32,
                    dest = "batchsize",
                    type = int,
                    help='minibatch size',
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

if options.resume:
    if options.outputDir is None:
        print >> sys.stderr, "must specify an output directory when resuming"
        sys.exit(1)

#----------

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
model = makeModel()

if options.cuda:
    model.cuda(options.cudaDevice)

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
logfname = os.path.join(options.outputDir, "train.log")
if os.path.exists(logfname):
    # append
    logfile = open(logfname, "a")
    print >> logfile,"----------------------------------------------------------------------"
else:
    logfile = open(logfname, "w")

fouts = [ sys.stdout, logfile ]

#----------

print "loading data"

doPtEtaReweighting = globals().get("doPtEtaReweighting", False)

# data e.g. to be shared between train and test dataset
datasetAuxData = {}

with Timer("loading training dataset...") as t:
    trainData, trsize = datasetLoadFunction(dataDesc['train_files'], options.size, 
                                            cuda = options.cuda, 
                                            isTraining = True,
                                            reweightPtEta = doPtEtaReweighting,
                                            logStreams = fouts,
                                            returnEventIds = False,
                                            auxData = datasetAuxData)
with Timer("loading test dataset...") as t:
    testData,  tesize = datasetLoadFunction(dataDesc['test_files'], options.size, options.cuda, 
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
# in our own model classes we have such a method,
# otherwise just assume that we have one output

if hasattr(model, 'getNumOutputNodes'):
    numOutputNodes = model.getNumOutputNodes()
else:
    numOutputNodes = 1

#----------

weightsTensor = torch.zeros(options.batchsize)
if options.cuda:
    weightsTensor = weightsTensor.cuda(options.cudaDevice)

# build the loss function and reshape
# the training weights tensor
#----------

if not globals().has_key('makeLoss'):
    makeLoss = makeDefaultLoss
    save_targets = False
else:
    # there was a custom loss function defined,
    # assume that we should also write out custom target
    # values (which are different from labels)
    # since this is likely a regression problem
    save_targets = True

lossFunc, trainWeights, testWeights = makeLoss(numOutputNodes, weightsTensor, trainData['weights'], testData['weights'])

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
    
for fout in fouts:
    print >> fout, "using",options.optimizer,"optimizer"

if lossFunc != None:
    if options.optimizer == 'adam':

        optimizer = optim.Adam(model.parameters(), lr = 0.0001)
    elif options.optimizer == 'sgd':

        # parameters taken from pyTorch Mnist example (?!)
        optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5)
    else:
        raise Exception("internal error")
else:
    optimizer = None

#----------
# load model and optimizer state if resuming
#----------

if options.resume:
    assert optimizer is not None
    epoch = loadCheckpoint(options.outputDir, model, optimizer)
else:
    epoch = 1

#----------
# convert targets to integers (needed for softmax)
#----------

if numOutputNodes != 1 and numOutputNodes != None:
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
    trainDataSet = makeDataSet(trainData)

with Timer("unpacking test dataset...", fouts) as t:
    testDataSet  = makeDataSet(testData)

train_output = np.zeros(len(trainData['labels']))

#----------
dataloader = DataLoader(trainDataSet, batch_size = options.batchsize, shuffle = True,
                        # TODO: add support for selecting a subset
                        # selectedIndices = selectedIndices
                        # num_workers = 4
                        )

#----------
# try to serialize the model structure itself
# will not work if used e.g. on CPU instead of GPU etc.
import cPickle as pickle
pickle.dump(
    dict(model = model,
         ), open(os.path.join(options.outputDir,
                                                     "model-structure.pkl"),"w"))
if hasattr(torch,'onnx'):
    dumpModelOnnx(model, os.path.join(options.outputDir, "model-structure.onnx"), options.cuda, dataloader)
#----------

print "params=",model.parameters()
print
print 'starting training at', time.asctime()


signal.signal(signal.SIGINT, breakHandler)

if options.pythonProfiling:
    profiler = cProfile.Profile()
    profiler.enable()

while True:

    #----------

    if options.maxEpochs != None and epoch > options.maxEpochs:
        break

    #----------

    epochIteration()

    #----------
    # checkpointing
    # see e.g. https://github.com/pytorch/examples/blob/7d0d413425e2ee64fcd0e0de1b11c5cca1f79f4d/imagenet/main.py#L165
    #----------

    state = dict(
            epoch = epoch,
            model_state = model.state_dict(),
            )

    if optimizer is None:
        state['optimizer_state'] = None
    else:
        state['optimizer_state'] = optimizer.state_dict()

    torch.save(state, os.path.join(options.outputDir, "checkpoint-%04d.torch" % epoch))

    #----------
    # prepare next iteration
    #----------
    epoch += 1

    if lossFunc == None:
        # just calculate the AUC of the training and test sample,
        # do not continue iterating
        break

    if stopFlag:
        break

# write out profiling data
if options.pythonProfiling:
    profiler.disable()
    outFname = os.path.join(options.outputDir, "pythonProfile.prof")
    profiler.dump_stats(outFname)

    # read stats back to print them and to print them to both stdout
    # and the log file

    import pstats
    for fout in fouts:
        stat = pstats.Stats(outFname, stream = fout).sort_stats('time').print_stats()

    print >> sys.stderr,"wrote profiling data to",outFname
