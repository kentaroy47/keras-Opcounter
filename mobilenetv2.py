# -*- coding: utf-8 -*-

import keras
from util import profile

log = True

# TODO: NotSupportedYet! == Depthwise conv
model = keras.applications.MobileNetV2(weights=None)

# look at model
model.summary()

# run profile
layer_name, layer_flops, inshape, weights = profile(model, log)

# visualize results
for name, flop, shape, weight in zip(layer_name, layer_flops, inshape, weights):
    print("layer:", name, shape, " MegaFLOPS:", flop/1e6, " MegaWeights:", weight/1e6)

print("Total FLOPS[GFLOPS]:", sum(layer_flops)/1e9)
