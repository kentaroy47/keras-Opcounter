# -*- coding: utf-8 -*-

import keras.backend as K
import numpy as np

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

def profile(model):
    # make lists
    layer_name = []
    layer_flops = []
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
            
    return layer_name, layer_flops, inshape, weights