#!/usr/bin/env python

# translation of https://github.com/facebookresearch/ResNeXt/blob/3cf474fdffa9ba4ce11ad41c0278e38fcd47372f/models/resnext.lua

from ConcatTable import ConcatTable

import torch.nn as nn
import torch
from torch.autograd import Variable


#----------------------------------------------------------------------

from Identity import Identity
from ReLU import ReLU
from Concat import Concat
from CAddTable import CAddTable
from Avg import Avg
from Marker import Marker
from View import View

Convolution = nn.Conv2d
Max = nn.MaxPool2d

# default arguments for eps and momentum match 
# between Torch/Lua and PyTorch
SBatchNorm = nn.BatchNorm2d

#----------------------------------------------------------------------

class ModelCreator:

    #----------------------------------------

    # the shortcut layer is either identity or 1x1 convolution
    def shortcut(self, nInputPlane, nOutputPlane, stride):
        useConv = self.shortcutType == 'C' or (self.shortcutType == 'B' and nInputPlane != nOutputPlane)
        if useConv:
            # 1x1 convolution
            return nn.Sequential(
                Convolution(nInputPlane, nOutputPlane, kernel_size = (1, 1), stride = stride, bias = False),
                SBatchNorm(nOutputPlane)
                )
        elif nInputPlane != nOutputPlane:
           # Strided, zero-padded identity shortcut
           return nn.Sequential(
               nn.SpatialAveragePooling(1, 1, stride, stride),
               Concat(1, [
                       Identity(),
                       nn.MulConstant(0)
                       ])
               )
        else:
           return Identity()

    #----------------------------------------

    # aggregated residual transformation bottleneck layer, Form (B)
    def split(self, nInputPlane, d, c, stride):
        cat = []
        for i in range(c):
            s = []
            s.append(Convolution(nInputPlane,d,kernel_size = (1,1), stride = (1,1), padding = (0,0), bias = False))
            s.append(SBatchNorm(d))
            s.append(ReLU())
            s.append(Convolution(d,d,kernel_size = (3,3), stride = stride, padding = (1,1), bias = False))
            s.append(SBatchNorm(d))
            s.append(ReLU())

            s = nn.Sequential(*s)
            cat.append(s)

        cat = ConcatTable(cat)

        return cat
    # end of function split()

    #----------------------------------------

    # original bottleneck residual layer
    def resnet_bottleneck(self, n, stride):

        nInputPlane = self.iChannels
        self.iChannels = n * 4

        s = nn.Sequential(
            Convolution(nInputPlane,n,kernel_size = (1,1), stride = (1,1), padding = (0,0), bias = False),
            SBatchNorm(n),
            ReLU(),
            Convolution(n,n,kernel_size = (3,3), stride = stride, padding = (1,1), bias = False),
            SBatchNorm(n),
            ReLU(),
            Convolution(n,n*4, kernel_size = (1,1), stride = (1,1), padding = (0,0), bias = False),
            SBatchNorm(n * 4),
            )
        
        return nn.Sequential(
            ConcatTable(
                [s,
                 self.shortcut(nInputPlane, n * 4, stride)]),
            # TODO: do this inplace
            CAddTable(),
            ReLU(),
            )

    # end of function resnet_bottleneck

    #----------------------------------------

    def resnext_bottleneck_B(self, n, stride):

        nInputPlane = self.iChannels
        self.iChannels = n * 4
  
        D = int(math.floor(n * (self.baseWidth/64.)))
        C = self.cardinality
  
        s = []
        s.append(split(nInputPlane, D, C, stride))
        s.append(JoinTable(2))
        s.append(Convolution(D*C,n*4,kernel_size = (1,1), stride = (1,1), padding = (0,0), bias = False))
        s.append(SBatchNorm(n*4))
  
        s = nn.Sequential(*s)

        return nn.Sequential(
            ConcatTable([
                    s,
                    self.shortcut(nInputPlane, n * 4, stride)]),
            CAddTable(true),
            ReLU())

    # end of function resnext_bottleneck_B
     
    #----------------------------------------

    def addMarker(self, moduleList, markerText):
        if self.addMarkers:
            moduleList.append(Marker(markerText))

    #----------------------------------------

    # aggregated residual transformation bottleneck layer, Form (C)
    def resnext_bottleneck_C(self, n, stride):

        nInputPlane = self.iChannels
        self.iChannels = n * 4
 
        import math
        D = int(math.floor(n * (self.baseWidth/64.)))
        C = self.cardinality
        
        s = []

        s.append(Convolution(nInputPlane,D*C,kernel_size = (1,1), stride = 1, padding = (0,0), bias = False))
        s.append(SBatchNorm(D*C))
        s.append(ReLU())
        s.append(Convolution(D*C,D*C,kernel_size = (3,3), stride = stride, padding = (1,1), groups = C, bias = False))
        s.append(SBatchNorm(D*C))
        s.append(ReLU())
        s.append(Convolution(D*C,n*4, kernel_size = (1,1), stride = 1, padding = (0,0), bias = False))
        s.append(SBatchNorm(n*4))
        
        s = nn.Sequential(*s)

        modules = []
        self.addMarker(modules, "begin resnet block C n=" + str(n) + " stride=" + str(stride))
        modules.extend([
                ConcatTable([
                        s,
                        self.shortcut(nInputPlane, n * 4, stride)]),
                CAddTable(),
                ReLU(),
                ]
           )
        self.addMarker(modules, "end resnet block C n=" + str(n) + " stride=" + str(stride))
        return nn.Sequential(*modules)

    # end of function resnext_bottleneck 

    #----------------------------------------

    # Creates count residual blocks with specified number of features
    def layer(self, block, features, count, stride):
        modules = [] 
        for i in range(count):
            
            if i == 0:
                modules.append(block(features, stride))
            else:
                modules.append(block(features, 1))

        return nn.Sequential(*modules)

    #----------------------------------------

    def ConvInit(self, name):
        
       for v in model.modules():
           n = v.kW*v.kH*v.nOutputPlane
           v.weight.normal(0,math.sqrt(2/float(n)))
           # if cudnn.version >= 4000 then
           #    v.bias = nil
           #    v.gradBias = nil
           # else
           v.bias.zero()

    # end of function ConvInit

    #----------------------------------------

    def __init__(self,
                 depth, 
                 shortcutType = 'B',
                 bottleneckType = None,
                 cardinality = None,
                 baseWidth = None,
                 dataset = None,
                 tensorType = torch.FloatTensor,
                 numInputPlanes = 3,
                 numOutputNodes = None,
                 avgKernelSize = None,
                 addMarkers = False,
                 ):

        assert shortcutType in ('A','B','C'), "unexpected shortcutType " + str(shortcutType)

        self.depth          = depth
        self.shortcutType   = shortcutType
        self.bottleneckType = bottleneckType
        self.cardinality    = cardinality
        self.baseWidth      = baseWidth   
        self.dataset        = dataset     
        self.tensorType     = tensorType
        self.numInputPlanes = numInputPlanes
        self.avgKernelSize  = avgKernelSize
        self.addMarkers     = addMarkers
        self.numOutputNodes = numOutputNodes

    #----------------------------------------

    def create(self):

        self.iChannels = None

        model = []

        if self.bottleneckType == 'resnet':
            bottleneck = self.resnet_bottleneck
            # print('Deploying ResNet bottleneck block')
        elif self.bottleneckType == 'resnext_B':
            bottleneck = self.resnext_bottleneck_B
            # print('Deploying ResNeXt bottleneck block form B')
        elif self.bottleneckType == 'resnext_C':
            bottleneck = self.resnext_bottleneck_C
            # print('Deploying ResNeXt bottleneck block form C (group convolution)')
        else:
            raise Exception('invalid bottleneck type: ' + str(self.bottleneckType))


        #----------

        if self.dataset == 'imagenet' :

            if self.avgKernelSize is None:
                self.avgKernelSize = 7

            # configurations for ResNet:
            #  num. residual blocks, num features, residual block function
            cfg = {
                # from the paper, table 1 left 
                # (nBlocks is the number after multiplication sign and 
                # big right closing bracket)
                50 : dict(nBlocks = (3, 4, 6, 3),  nFeatures = 2048, block = bottleneck),

               101 : dict(nBlocks = (3, 4, 23, 3), nFeatures = 2048, block = bottleneck),
               152 : dict(nBlocks = (3, 8, 36, 3), nFeatures = 2048, block = bottleneck),
            }

            assert cfg.has_key(self.depth), 'Invalid depth: ' + str(self.depth)
            nBlocks, nFeatures, block = cfg[self.depth]['nBlocks'], cfg[self.depth]['nFeatures'], cfg[self.depth]['block']
            self.iChannels = 64
            # print(' | ResNet-' .. self.depth .. ' ImageNet')

            # ResNet ImageNet model

            # stage conv1
            model.append(Convolution(self.numInputPlanes,64,kernel_size = (7,7), stride = 2, padding = (3,3), bias = False))
            model.append(SBatchNorm(64))
            model.append(ReLU())

            # stage conv2
            self.addMarker(model, "begin stage conv2")
            model.append(Max(kernel_size = (3,3), stride = (2,2), padding = (1,1)))
            model.append(self.layer(block, features =  64, count = nBlocks[0], stride = 1))
            self.addMarker(model, "end stage conv2")

            # stage conv3
            self.addMarker(model, "begin stage conv3")
            model.append(self.layer(block, features = 128, count = nBlocks[1], stride = 2))
            self.addMarker(model, "end stage conv3")

            # stage conv4
            self.addMarker(model,"begin stage conv4")
            model.append(self.layer(block, features = 256, count = nBlocks[2], stride = 2))
            self.addMarker(model, "end stage conv4")

            # stage conv5
            self.addMarker(model,"begin stage conv5")
            model.append(self.layer(block, features = 512, count = nBlocks[3], stride = 2))
            self.addMarker(model, "end stage conv5")

            model.append(Avg(kernel_size = self.avgKernelSize, stride = (1, 1)))

            # see also https://github.com/torch/nn/blob/master/doc/simple.md#nn.View
            # model.append(nn.View(nFeatures):setNumInputDims(3))

            model.append(View(nFeatures))

            if self.numOutputNodes is None:
                self.numOutputNodes = 1000

            model.append(nn.Linear(nFeatures, self.numOutputNodes))

        elif self.dataset == 'cifar10' or self.dataset == 'cifar100':

            if self.avgKernelSize is None:
                self.avgKernelSize = 8

            # model type specifies number of layers for CIFAR-10 and CIFAR-100 model
            assert (self.depth - 2) % 9 == 0, 'depth should be one of 29, 38, 47, 56, 101'
            n = (self.depth - 2) / 9

            self.iChannels = 64
            # print(' | ResNet-' .. depth .. ' ' .. opt.dataset)

            model.append(Convolution(self.numInputPlanes,64,kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = False))
            model.append(SBatchNorm(64))
            model.append(ReLU())
            self.addMarker(model, "begin layer 1")
            model.append(self.layer(bottleneck, 64, n, 1))
            self.addMarker(model, "end layer 1")

            self.addMarker(model, "begin layer 2")
            model.append(self.layer(bottleneck, 128, n, 2))
            self.addMarker(model, "end layer 2")

            self.addMarker(model, "begin layer 3")
            model.append(self.layer(bottleneck, 256, n, 2))
            self.addMarker(model, "end layer 3")

            model.append(Avg(kernel_size = self.avgKernelSize, stride = (1, 1)))

            # model.append(nn.View(1024):setNumInputDims(3))
            model.append(View(1024))

            if self.numOutputNodes is None:
                if self.dataset == 'cifar10':
                    self.numOutputNodes = 10
                else: 
                    self.numOutputNodes = 100
            model.append(nn.Linear(1024, self.numOutputNodes))
        else:
            raise Exception('invalid dataset: ' + str(dataset))

        #----------

        model = nn.Sequential(*model)

        model.type(self.tensorType)

        #----------
        # initialization
        #----------

        def initFunc(module):
            if isinstance(module, nn.modules.conv.Conv2d):
                nn.init.kaiming_normal(module.weight)
            elif isinstance(module, nn.modules.batchnorm.BatchNorm2d):
                module.weight.data[...] = 1
                module.bias.data[...] = 0
            elif isinstance(module, nn.modules.linear.Linear):
                module.bias.data[...] = 0
                nn.init.kaiming_normal(module.weight)

        model.apply(initFunc)

        return model
    # end of function create()
 
#----------------------------------------------------------------------

if __name__ == '__main__':

    dataset, mode = 'cifar10', 'ecal'

    if mode == 'ecal':
        numInputPlanes = 1
        avgKernelSize = 9
    else:
        numInputPlanes = 3
        avgKernelSize = None

    # ResNeXt 16x64d for CIFAR10
    # parameters from command line arguments
    # at https://github.com/facebookresearch/ResNeXt#1x-complexity-configurations-reference-table
    model = ModelCreator(depth = 29, 
                         cardinality = 16,
                         baseWidth = 64,

                         dataset = dataset,
                         bottleneckType = 'resnext_C',
                         numInputPlanes = numInputPlanes,
                         avgKernelSize = avgKernelSize,
                         addMarkers = True,
                        ).create()
    print model

    print "----------------------------------------------------------------------"
    import sys
    sys.stderr.flush()
    sys.stdout.flush()

    # ImageNet
    # inputVar = Variable(torch.FloatTensor(32,3,224,224), requires_grad = False)

    # CIFAR

    if mode == 'cifar10':
        inputVar = Variable(torch.FloatTensor(32,3,32,32), requires_grad = False)

    elif mode == 'ecal':
        # our own ECAL images
        inputVar = Variable(torch.FloatTensor(32,1,35,35), requires_grad = False)

    from PrintHook import PrintHook

    printHook = PrintHook()
    printHook.registerWith(model)

    # check: forward pass
    print "forwarding"
    model(inputVar)
    print "done forwarding"

    if hasattr(torch,'onnx'):

        torch.onnx.export(model,
                          args = inputVar,
                          f = "/tmp/resnext.onnx",
                          # also write weights out so that
                          # we get the initializer fields
                          # and can distinguish between
                          # variable inputs and trained weights.
                          #
                          # (even if the model is not trained
                          # at the beginning)
                          export_params = True,
                          )
    else:
        print >> sys.stderr,"this installation of torch does not have onnx support"
