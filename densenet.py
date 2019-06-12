# -*- coding: utf-8 -*-
"""
add your model discription and calculate the computation!
"""

import keras
from util import profile

log = True

model = keras.applications.densenet.DenseNet121(weights=None)

# look at model
model.summary()

# run profile
layer_name, layer_flops, inshape, weights = profile(model, log)

# visualize results
for name, flop, shape, weight in zip(layer_name, layer_flops, inshape, weights):
    print("layer:", name, shape, " MegaFLOPS:", flop/1e6, " MegaWeights:", weight/1e6)

print("Total FLOPS[GFLOPS]:", sum(layer_flops)/1e9)
