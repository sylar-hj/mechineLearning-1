from time import sleep
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

#
# ''' 1 '''
#
# i = 0
# for index in range(100):
#     if i > 15:
#         i = 0
#         plt.figure()
#
#     plt.subplot(4, 4,i + 1)
#     plt.title('test')
#     i += 1
# plt.show()


# ''' 2 '''
#
# a = [1, 2, 3]
# print(a[-1])

#
# ''' 3 '''
#
# print(1*\
#       2)


# """ 4 """
# class Network(object):
#     def __init__(self, sizes):
#         """The list ``sizes`` contains the number of neurons in the
#         respective layers of the network.  For example, if the list
#         was [2, 3, 1] then it would be a three-layer network, with the
#         first layer containing 2 neurons, the second layer 3 neurons,
#         and the third layer 1 neuron.  The biases and weights for the
#         network are initialized randomly, using a Gaussian
#         distribution with mean 0, and variance 1.  Note that the first
#         layer is assumed to be an input layer, and by convention we
#         won't set any biases for those neurons, since biases are only
#         ever used in computing the outputs from later layers."""
#         self.num_layers = len(sizes)
#         self.sizes = sizes
#         self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
#         self.weights = [np.random.randn(y, x)
#                         for x, y in zip(sizes[:-1], sizes[1:])]
#
#         nabla_w = [np.zeros(w.shape) for w in self.weights]
#         print(nabla_w)
#         print(list(zip(self.weights, nabla_w)))
#
#
# Network([3, 2, 1])

""" 5 """
a = [1, 2, 3]
a.pop()
print(a)