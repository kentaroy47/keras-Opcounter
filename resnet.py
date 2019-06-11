# -*- coding: utf-8 -*-
"""
add your model discription and calculate the computation!
"""

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D
from keras.engine.input_layer import Input
import keras

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
    # number of conv operations = input_h * input_w / stride
    try:
        numshifts = int(layers.input_shape[1] * layers.input_shape[2] / layers.get_config()["strides"][0])
    except:
        numshifts = int(layers.input_shape[1] * layers.input_shape[2]) # for zeropadding2D
        
    # MAC/convfilter = kernelsize^2 * InputChannels * OutputChannels
    MACperConv = layers.get_config()["kernel_size"][0] * layers.get_config()["kernel_size"][1] * layers.input_shape[3] * layers.output_shape[3]
    
    if layers.get_config()["use_bias"]:
        ADD = layers.output_shape[1] * layers.output_shape[2] * layers.output_shape[3]
    else:
        ADD = 0
        
    return MACperConv * numshifts * 2 + ADD

# TODO: RELUs
# TODO: BatchNorms
# Residual Paths

# run through models
for layer in model.layers:
    
    if "dense" in layer.get_config()["name"] or "fc" in layer.get_config()["name"]:
        layer_flops.append(count_linear(layer))
        layer_name.append(layer.get_config()["name"])
    elif "conv" in layer.get_config()["name"] and "pad" not in layer.get_config()["name"] and "bn" not in layer.get_config()["name"]:
        layer_flops.append(count_conv2d(layer))
        layer_name.append(layer.get_config()["name"])
    elif "res" in layer.get_config()["name"] and "branch" in layer.get_config()["name"]:
        layer_flops.append(count_conv2d(layer))
        layer_name.append(layer.get_config()["name"])
        
model.summary()
        
print("layers:", layer_name)
print("FLOPS:", layer_flops)

print("Total FLOPS[GFLOPS]:", sum(layer_flops)/10e9)

# TODO: summarize results as dict