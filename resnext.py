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

iChannels = None

def createModel(depth, shortcutType = 'B',
                bottleneckType = None,
                cardinality = None,
                baseWidth = None,
                dataset = None,
                tensorType = torch.FloatTensor,
                numInputPlanes = 3
                ):

    assert shortcutType in ('A','B','C'), "unexpected shortcutType " + str(shortcutType)

    #----------

    # the shortcut layer is either identity or 1x1 convolution
    def shortcut(nInputPlane, nOutputPlane, stride):
        useConv = shortcutType == 'C' or (shortcutType == 'B' and nInputPlane != nOutputPlane)
        if useConv:
            # 1x1 convolution
            return nn.Sequential(
                Convolution(nInputPlane, nOutputPlane, kernel_size = (1, 1), stride = stride),
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

    #----------

    
    # original bottleneck residual layer
    def resnet_bottleneck(n, stride):

        global iChannels
        nInputPlane = iChannels
        iChannels = n * 4

        s = nn.Sequential(
            Convolution(nInputPlane,n,kernel_size = (1,1), stride = (1,1), padding = (0,0)),
            SBatchNorm(n),
            ReLU(),
            Convolution(n,n,kernel_size = (3,3), stride = stride, padding = (1,1)),
            SBatchNorm(n),
            ReLU(),
            Convolution(n,n*4, kernel_size = (1,1), stride = (1,1), padding = (0,0)),
            SBatchNorm(n * 4),
            )
        
        return nn.Sequential(
            ConcatTable(
                [s,
                 shortcut(nInputPlane, n * 4, stride)]),
            # TODO: do this inplace
            CAddTable(),
            ReLU(),
            )

    # end of function resnet_bottleneck

    #----------
   
    # aggregated residual transformation bottleneck layer, Form (B)
    def split(nInputPlane, d, c, stride):
        cat = []
        for i in range(c):
            s = []
            s.append(Convolution(nInputPlane,d,kernel_size = (1,1), stride = (1,1), padding = (0,0)))
            s.append(SBatchNorm(d))
            s.append(ReLU())
            s.append(Convolution(d,d,kernel_size = (3,3), stride = stride, padding = (1,1)))
            s.append(SBatchNorm(d))
            s.append(ReLU())

            s = nn.Sequential(*s)
            cat.append(s)

        cat = ConcatTable(cat)

        return cat
    # end of function split()

    #----------
    
    def resnext_bottleneck_B(n, stride):
        global iChannels
        nInputPlane = iChannels
        iChannels = n * 4
  
        D = int(math.floor(n * (opt.baseWidth/64.)))
        C = cardinality
  
        s = []
        s.append(split(nInputPlane, D, C, stride))
        s.append(JoinTable(2))
        s.append(Convolution(D*C,n*4,kernel_size = (1,1), stride = (1,1), padding = (0,0)))
        s.append(SBatchNorm(n*4))
  
        s = nn.Sequential(*s)

        return nn.Sequential(
            ConcatTable([
                    s,
                    shortcut(nInputPlane, n * 4, stride)]),
            CAddTable(true),
            ReLU())

    # end of function resnext_bottleneck_B
     
    #----------

    # aggregated residual transformation bottleneck layer, Form (C)
    def resnext_bottleneck_C(n, stride = None):
        global iChannels
        nInputPlane = iChannels

        iChannels = n * 4
 
        import math
        D = int(math.floor(n * (baseWidth/64.)))
        C = cardinality
        
        s = []

        s.append(Convolution(nInputPlane,D*C,kernel_size = (1,1), stride = 1, padding = (0,0)))
        s.append(SBatchNorm(D*C))
        s.append(ReLU())
        s.append(Convolution(D*C,D*C,kernel_size = (3,3), stride = stride, padding = (1,1), groups = C))
        s.append(SBatchNorm(D*C))
        s.append(ReLU())
        s.append(Convolution(D*C,n*4, kernel_size = (1,1), stride = 1, padding = (0,0)))
        s.append(SBatchNorm(n*4))
        
        s = nn.Sequential(*s)
 
        return nn.Sequential(
            Marker("begin resnet block C n=" + str(n) + " stride=" + str(stride)),

            ConcatTable([
                    s,
                    shortcut(nInputPlane, n * 4, stride)]),
            CAddTable(),
           ReLU(),
           Marker("end resnet block C n=" + str(n) + " stride=" + str(stride)),
           )
    # end of function resnext_bottleneck 

    #----------

    # Creates count residual blocks with specified number of features
    def layer(block, features, count, stride):
        modules = [] 
        for i in range(count):
            
            if i == 0:
                modules.append(block(features, stride))
            else:
                modules.append(block(features, 1))

        return nn.Sequential(*modules)

    #----------

 
    model = []

    if bottleneckType == 'resnet':
        bottleneck = resnet_bottleneck
        # print('Deploying ResNet bottleneck block')
    elif bottleneckType == 'resnext_B':
        bottleneck = resnext_bottleneck_B
        # print('Deploying ResNeXt bottleneck block form B')
    elif bottleneckType == 'resnext_C':
        bottleneck = resnext_bottleneck_C
        # print('Deploying ResNeXt bottleneck block form C (group convolution)')
    else:
        raise Exception('invalid bottleneck type: ' + str(bottleneckType))

    
    #----------

    global iChannels
    if dataset == 'imagenet' :
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
        
        assert cfg.has_key(depth), 'Invalid depth: ' + str(depth)
        nBlocks, nFeatures, block = cfg[depth]['nBlocks'], cfg[depth]['nFeatures'], cfg[depth]['block']
        iChannels = 64
        # print(' | ResNet-' .. depth .. ' ImageNet')
        
        # ResNet ImageNet model

        # stage conv1
        model.append(Convolution(numInputPlanes,64,kernel_size = (7,7), stride = 2, padding = (3,3)))
        model.append(SBatchNorm(64))
        model.append(ReLU())

        # stage conv2
        model.append(Marker("begin stage conv2"))
        model.append(Max(kernel_size = (3,3), stride = (2,2), padding = (1,1)))
        model.append(layer(block, features =  64, count = nBlocks[0], stride = 1))
        model.append(Marker("end stage conv2"))

        # stage conv3
        model.append(Marker("begin stage conv3"))
        model.append(layer(block, features = 128, count = nBlocks[1], stride = 2))
        model.append(Marker("end stage conv3"))

        # stage conv4
        model.append(Marker("begin stage conv4"))
        model.append(layer(block, features = 256, count = nBlocks[2], stride = 2))
        model.append(Marker("end stage conv4"))

        # stage conv5
        model.append(Marker("begin stage conv5"))
        model.append(layer(block, features = 512, count = nBlocks[3], stride = 2))
        model.append(Marker("end stage conv5"))

        model.append(Avg(kernel_size = (7, 7), stride = (1, 1)))

        # see also https://github.com/torch/nn/blob/master/doc/simple.md#nn.View
        # model.append(nn.View(nFeatures):setNumInputDims(3))
        
        model.append(View(nFeatures))

        model.append(nn.Linear(nFeatures, 1000))
    
    elif dataset == 'cifar10' or dataset == 'cifar100':

        # model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 9 == 0, 'depth should be one of 29, 38, 47, 56, 101'
        n = (depth - 2) / 9
        
        iChannels = 64
        # print(' | ResNet-' .. depth .. ' ' .. opt.dataset)
  
        model.append(Convolution(numInputPlanes,64,kernel_size = (3,3), stride = (1,1), padding = (1,1)))
        model.append(SBatchNorm(64))
        model.append(ReLU())
        model.append(Marker("begin layer 1"))
        model.append(layer(bottleneck, 64, n, 1))
        model.append(Marker("end layer 1"))

        model.append(Marker("begin layer 2"))
        model.append(layer(bottleneck, 128, n, 2))
        model.append(Marker("end layer 2"))

        model.append(Marker("begin layer 3"))
        model.append(layer(bottleneck, 256, n, 2))
        model.append(Marker("end layer 3"))

        model.append(Avg(kernel_size = (8, 8), stride = (1, 1)))

        # model.append(nn.View(1024):setNumInputDims(3))
        model.append(View(1024))

        
        if dataset == 'cifar10':
            nCategories = 10  
        else: 
            nCategories = 100
        model.append(nn.Linear(1024, nCategories))
    else:
        raise Exception('invalid dataset: ' + str(dataset))

    #----------
 
    def ConvInit(name):
        
       for v in model.modules():
           n = v.kW*v.kH*v.nOutputPlane
           v.weight.normal(0,math.sqrt(2/float(n)))
           # if cudnn.version >= 4000 then
           #    v.bias = nil
           #    v.gradBias = nil
           # else
           v.bias.zero()

    # end of function ConvInit

    #----------

    model = nn.Sequential(*model)

    model.type(tensorType)
 
    # if cudnn == 'deterministic':
    #     def helper(m):
    #         if hasattr(m,'setMode'):
    #             m.setMode(1,1,1)
    # 
    #     model.apply(helper)
 
    # model.modules()[0].gradInput = None
 
    return model
# end of function createModel()
 
#----------------------------------------------------------------------

if __name__ == '__main__':

    # ResNeXt 16x64d for CIFAR10
    # parameters from command line arguments
    # at https://github.com/facebookresearch/ResNeXt#1x-complexity-configurations-reference-table
    model = createModel(depth = 29, 
                        cardinality = 16,
                        baseWidth = 64,

                        dataset = 'cifar10',
                        bottleneckType = 'resnext_C',

                        )
    print model

    print "----------------------------------------------------------------------"
    import sys
    sys.stderr.flush()
    sys.stdout.flush()

    # ImageNet
    # inputVar = Variable(torch.FloatTensor(32,3,224,224), requires_grad = False)

    # CIFAR
    inputVar = Variable(torch.FloatTensor(32,3,32,32), requires_grad = False)


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
