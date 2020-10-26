# -*- coding: utf-8 -*-
"""
add your model discription and calculate the computation!
"""

import keras
import kerop

# TODO: RELUs
# TODO: BatchNorms
# Residual Paths
model = keras.applications.resnet50.ResNet50(weights=None)

# look at model
model.summary()

# run profile
layer_name, layer_flops, inshape, weights = kerop.profile(model)

# visualize results
for name, flop, shape, weight in zip(layer_name, layer_flops, inshape, weights):
    print("layer:", name, shape, " MegaFLOPS:", flop/1e6, " MegaWeights:", weight/1e6)

print("Total FLOPS[GFLOPS]:", sum(layer_flops)/1e9)


# TODO: summarize results as dict