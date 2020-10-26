# -*- coding: utf-8 -*-
"""
add your model discription and calculate the computation!
"""

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D
from keras.engine.input_layer import Input
from kerop import profile

# write your model here!
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), input_shape=(28, 28, 1)))
model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1)))
model.add(Flatten())
model.add(Dense(320,))
model.add(Dense(100, activation='softmax'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# look at model
model.summary()

# run profile
layer_name, layer_flops, inshape, weights = profile(model)

# visualize results
for name, flop, shape, weight in zip(layer_name, layer_flops, inshape, weights):
    print("layer:", name, shape, " MegaFLOPS:", flop/1e6, " MegaWeights:", weight/1e6)

print("Total FLOPS[GFLOPS]:", sum(layer_flops)/1e9)

# TODO: summarize as dict