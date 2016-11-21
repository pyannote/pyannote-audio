#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Grégory GELLY
# Hervé BREDIN - http://herve.niderb.fr

import numpy as np


def unitary_angular_triplet_loss(anchor, positive, negative):
    epsilon = 1e-6

    dotProdPosAnc = np.clip(np.sum(positive*anchor), -1.0, 1.0)
    dotProdNegAnc = np.clip(np.sum(negative*anchor), -1.0, 1.0)

    localCost = (np.arccos(dotProdPosAnc)-np.arccos(dotProdNegAnc)-np.pi/60.0)
    coeffSlope = 1.0
    coeffSlopeNegative = 1.0
    if (localCost < 0.0):
        coeffSlope = coeffSlopeNegative
    coeffSlopeInternal = 10.0
    localCost *= coeffSlopeInternal
    localCost = 1.0/(1.0 + np.exp(-localCost))

    dotProdPosAnc = 1-dotProdPosAnc*dotProdPosAnc
    dotProdNegAnc = 1-dotProdNegAnc*dotProdNegAnc
    if (dotProdPosAnc < epsilon): dotProdPosAnc = epsilon
    if (dotProdNegAnc < epsilon): dotProdNegAnc = epsilon

    derivCoeff = localCost*(1.0-localCost)*coeffSlope*coeffSlopeInternal
    localCost = coeffSlope*localCost+(coeffSlopeNegative-coeffSlope)*0.5

    derivativeAnchor = (-positive/np.sqrt(dotProdPosAnc)+negative/np.sqrt(dotProdNegAnc))*derivCoeff
    derivativePositive = -anchor/np.sqrt(dotProdPosAnc)*derivCoeff
    derivativeNegative = (anchor/np.sqrt(dotProdNegAnc))*derivCoeff

    return [localCost, derivativeAnchor, derivativePositive, derivativeNegative]

def unitary_cosine_triplet_loss(anchor, positive, negative):
    dotProdPosAnc = np.sum(positive*anchor)
    dotProdNegAnc = np.sum(negative*anchor)

    localCost = -dotProdPosAnc+dotProdNegAnc-1.0/30.0
    localCost = 1.0/(1.0 + np.exp(-localCost))

    derivCoeff = 1.0
    derivativeAnchor = (-positive+negative)*derivCoeff
    derivativePositive = -anchor*derivCoeff
    derivativeNegative = anchor*derivCoeff

    return [localCost, derivativeAnchor, derivativePositive, derivativeNegative]

def unitary_euclidean_triplet_loss(anchor, positive, negative):

    localCost = np.sum(np.square(positive-anchor))-np.sum(np.square(negative-anchor))+0.2
    localCost = 1.0/(1.0 + np.exp(-localCost))

    derivativeAnchor = -2.0*(positive-negative)
    derivativePositive = -2.0*(anchor-positive)
    derivativeNegative = -2.0*(negative-anchor)

    return [localCost, derivativeAnchor, derivativePositive, derivativeNegative]
