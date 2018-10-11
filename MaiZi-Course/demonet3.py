# -*- coding: utf-8-*- #
import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
from conv import mini_batch_size

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv, softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample


def ReLU(z): return T.maximum(0, z)


training_data, validation_data, test_data = network3.load_data_shared('data/mnist.pkl.gz')
mini_batch_size = 10
expanded_training_data, _, _ = network3.load_data_shared("data/mnist_expanded.pkl.gz")

net = Network([
    ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                  filter_shape=(20, 1, 5, 5),
                  poolsize=(2, 2),
                  activation_fn=ReLU),
    ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                  filter_shape=(40, 20, 5, 5),
                  poolsize=(2, 2),
                  activation_fn=ReLU),
    FullyConnectedLayer(
        n_in=40 * 4 * 4, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
    FullyConnectedLayer(
        n_in=1000, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
    SoftmaxLayer(n_in=1000, n_out=10, p_dropout=0.5)],
    mini_batch_size)

net.SGD(expanded_training_data, 40, mini_batch_size,
        0.03, validation_data, test_data)
