# keras-Opcounter
calculate number of OPS in a Keras model!

right now, supports conv2d and dense only.

# Usage

## Count the ops in VGG16:

```
python vgg.py
```

Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0
_________________________________________________________________
layers: ['input_5', 'block1_conv1', 'block1_conv2', 'block1_pool', 'block2_conv1', 'block2_conv2', 'block2_pool', 'block3_conv1', 'block3_conv2', 'block3_conv3', 'block3_pool', 'block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_pool', 'block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_pool', 'flatten', 'fc1', 'fc2', 'predictions']
FLOPS: [176619520, 3702587392, 1851293696, 3700981760, 1850490880, 3700178944, 3700178944, 1850089472, 3699777536, 3699777536, 924944384, 924944384, 924944384, 205524992, 33558528]

Total FLOPS[GFLOPS]: 3.0945892352

## Count the ops in resnet50:

```
python resnet.py
```
