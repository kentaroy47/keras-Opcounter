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

# call Resnet50
model = keras.applications.resnet50.ResNet50()

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
    print(layers.get_config())
    print(layers.input_shape)
    print(layers.output_shape)
    # number of conv operations = input_h * input_w / stride
    try:
        numshifts = int(layers.input_shape[1] * layers.input_shape[2] / layers.get_config()["strides"][0])
    except:
        numshifts = int(layers.input_shape[1] * layers.input_shape[2]) # for zeropadding2D
        
    # MAC/convfilter = kernelsize^2 * InputChannels * OutputChannels
    MACperConv = layers.get_config()["kernel_size"][0] * layers.get_config()["kernel_size"][1] * layers.input_shape[3] * layers.output_shape[3]
    
    if layers.get_config()["use_bias"]:
        ADD = layers.output_shape[3]
    else:
        ADD = 0
        
    return MACperConv * numshifts * 2 + ADD

# TODO: RELUs
# TODO: BatchNorms
# Residual Paths

inshape = []
weights = []
# run through models
for layer in model.layers:
    
    if "dense" in layer.get_config()["name"] or "fc" in layer.get_config()["name"]:
        layer_flops.append(count_linear(layer))
        layer_name.append(layer.get_config()["name"])
        inshape.append(layer.input_shape)
        weights.append(int(np.sum([K.count_params(p) for p in set(layer.trainable_weights)])))
    elif "conv" in layer.get_config()["name"] and "pad" not in layer.get_config()["name"] and "bn" not in layer.get_config()["name"]:
        layer_flops.append(count_conv2d(layer))
        layer_name.append(layer.get_config()["name"])
        inshape.append(layer.input_shape)
        weights.append(int(np.sum([K.count_params(p) for p in set(layer.trainable_weights)])))
    elif "res" in layer.get_config()["name"] and "branch" in layer.get_config()["name"]:
        layer_flops.append(count_conv2d(layer))
        layer_name.append(layer.get_config()["name"])
        inshape.append(layer.input_shape)
        weights.append(int(np.sum([K.count_params(p) for p in set(layer.trainable_weights)])))
        
model.summary()
        
for name, flop, shape, weight in zip(layer_name, layer_flops, inshape, weights):
    print("layer:", name, shape, " MegaFLOPS:", flop/1e6, " MegaWeights:", weight/1e6)

print("Total FLOPS[GFLOPS]:", sum(layer_flops)/1e9)

# TODO: summarize results as dict