#!/usr/bin/env python

import numpy as np
#----------------------------------------------------------------------

# taken from the Lasagne mnist example and modified
def iterate_minibatches(targets, batchsize, shuffle = False, selectedIndices = None):
    # generates list of indices and target values

    if not selectedIndices is None:
        # a subset of indices was specified
        # make a copy
        indices = list(selectedIndices)
    else:
        indices = np.arange(len(targets))

    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, len(indices) - batchsize + 1, batchsize):

        excerpt = indices[start_idx:start_idx + batchsize]

        yield excerpt, targets[excerpt]

import time

#----------------------------------------------------------------------

# my own gradient update update supporting learning rate decay
# like we use in Torch)

# torch learning rate decay is implemented here:
# https://github.com/torch/optim/blob/b812d2a381162bed9f0df26cab8abb4015f47471/sgd.lua#L27
#
# see other Lasagne code e.g. here: https://github.com/Lasagne/Lasagne/blob/996bf64c0aec6d481044495800b461cc62040041/lasagne/updates.py#L584

def sgdWithLearningRateDecay(loss_or_grads, params, learningRate, learningRateDecay):

    from lasagne.updates import get_or_compute_grads

    import theano.tensor as T
    import theano

    from collections import OrderedDict
    from lasagne import utils

    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    t_prev = theano.shared(utils.floatX(0.))
    one = T.constant(1)

    t = t_prev + 1

    clr = learningRate / (1 + t * learningRateDecay)

    # http://leon.bottou.org/publications/pdf/tricks-2012.pdf
    # for example suggests (section 5.2)
    # "use learning rates of the form 
    #  gamma_t = gamma_0 / (1 + gamma_0 * lambda * t)
    # determine the best gamma_0 using a small training 
    # data sample" 
    # (lambda / 2 is the coefficient of the weights norm
    #  of L2 regularization)

    for param, grad in zip(params, grads):
        updates[param] = param - clr * grad

    updates[t_prev] = t

    return updates

#----------------------------------------------------------------------
