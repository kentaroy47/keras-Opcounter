# -*- coding: utf-8 -*-
"""
add your model discription and calculate the computation!
"""

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D
from keras.engine.input_layer import Input
import keras
import keras.backend as K
import numpy as np

# VGG
model = keras.applications.vgg16.VGG16()

# make lists
layer_name = []
layer_flops = []
layer_params = []

# bunch of cal per layer
def count_linear(layers):
    MAC = layers.output_shape[1] * layers.input_shape[1]
    if layers.get_config()["use_bias"]:
        ADD = layers.output_shape[1]
    else:
        ADD = 0
    return MAC*2 + ADD

def count_conv2d(layers):
    # number of conv operations = input_h * input_w / stride
    numshifts = int(layers.input_shape[1] * layers.input_shape[2] / layers.get_config()["strides"][0])
    
    # MAC/convfilter = kernelsize^2 * InputChannels * OutputChannels
    MACperConv = layers.get_config()["kernel_size"][0] * layers.get_config()["kernel_size"][1] * layers.input_shape[3] * layers.output_shape[3]
    
    if layers.get_config()["use_bias"]:
        ADD = layers.output_shape[3]
    else:
        ADD = 0
        
    return MACperConv * numshifts * 2 + ADD

# TODO: relus

inshape = []
weights = []
# run through models
for layer in model.layers:
    if "dense" in layer.get_config()["name"] or "fc" in layer.get_config()["name"]:
        layer_flops.append(count_linear(layer))
        layer_name.append(layer.get_config()["name"])
        inshape.append(layer.input_shape)
        weights.append(int(np.sum([K.count_params(p) for p in set(layer.trainable_weights)])))
    elif "conv" in layer.get_config()["name"]:
        layer_flops.append(count_conv2d(layer))
        layer_name.append(layer.get_config()["name"])
        inshape.append(layer.input_shape)
        weights.append(int(np.sum([K.count_params(p) for p in set(layer.trainable_weights)])))
        
model.summary()

for name, flop, shape, weight in zip(layer_name, layer_flops, inshape, weights):
    print("layer:", name, shape, " MegaFLOPS:", flop/1e6, " MegaWeights:", weight/1e6)

print("Total FLOPS[GFLOPS]:", sum(layer_flops)/1e9)

# TODO: summarize results as dict

"""
show in this formatINPUT:     [224x224x3]    memory:  224*224*3=150K   weights: 0
CONV3-64:  [224x224x64]   memory:  224*224*64=3.2M  weights: (3*3*3)*64 = 1,728
CONV3-64:  [224x224x64]   memory:  224*224*64=3.2M  weights: (3*3*64)*64 = 36,864
POOL2:     [112x112x64]   memory:  112*112*64=800K  weights: 0
CONV3-128: [112x112x128]  memory:  112*112*128=1.6M weights: (3*3*64)*128 = 73,728
CONV3-128: [112x112x128]  memory:  112*112*128=1.6M weights: (3*3*128)*128 = 147,456
POOL2:     [56x56x128]    memory:  56*56*128=400K   weights: 0
CONV3-256: [56x56x256]    memory:  56*56*256=800K   weights: (3*3*128)*256 = 294,912
CONV3-256: [56x56x256]    memory:  56*56*256=800K   weights: (3*3*256)*256 = 589,824
CONV3-256: [56x56x256]    memory:  56*56*256=800K   weights: (3*3*256)*256 = 589,824
POOL2:     [28x28x256]    memory:  28*28*256=200K   weights: 0
CONV3-512: [28x28x512]    memory:  28*28*512=400K   weights: (3*3*256)*512 = 1,179,648
CONV3-512: [28x28x512]    memory:  28*28*512=400K   weights: (3*3*512)*512 = 2,359,296
CONV3-512: [28x28x512]    memory:  28*28*512=400K   weights: (3*3*512)*512 = 2,359,296
POOL2:     [14x14x512]    memory:  14*14*512=100K   weights: 0
CONV3-512: [14x14x512]    memory:  14*14*512=100K   weights: (3*3*512)*512 = 2,359,296
CONV3-512: [14x14x512]    memory:  14*14*512=100K   weights: (3*3*512)*512 = 2,359,296
CONV3-512: [14x14x512]    memory:  14*14*512=100K   weights: (3*3*512)*512 = 2,359,296
POOL2:     [7x7x512]      memory:  7*7*512=25K      weights: 0
FC:        [1x1x4096]     memory:  4096             weights: 7*7*512*4096 = 102,760,448
FC:        [1x1x4096]     memory:  4096             weights: 4096*4096 = 16,777,216
FC:        [1x1x1000]     memory:  1000             weights: 4096*1000 = 4,096,000

total memory
total weights
total computations
"""
