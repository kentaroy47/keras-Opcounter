# keras-Opcounter
calculate number of OPS in a Keras model!

## still under development!
Right now, supports conv2d and dense only.
Doesn't fully count the activations yet.

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
Total params: 25,636,712
Trainable params: 25,583,592
Non-trainable params: 53,120
__________________________________________________________________________________________________
layers: ['conv1', 'res2a_branch2a', 'res2a_branch2b', 'res2a_branch2c', 'res2a_branch1', 'res2b_branch2a', 'res2b_branch2b', 'res2b_branch2c', 'res2c_branch2a', 'res2c_branch2b', 'res2c_branch2c', 'res3a_branch2a', 'res3a_branch2b', 'res3a_branch2c', 'res3a_branch1', 'res3b_branch2a', 'res3b_branch2b', 'res3b_branch2c', 'res3c_branch2a', 'res3c_branch2b', 'res3c_branch2c', 'res3d_branch2a', 'res3d_branch2b', 'res3d_branch2c', 'res4a_branch2a', 'res4a_branch2b', 'res4a_branch2c', 'res4a_branch1', 'res4b_branch2a', 'res4b_branch2b', 'res4b_branch2c', 'res4c_branch2a', 'res4c_branch2b', 'res4c_branch2c', 'res4d_branch2a', 'res4d_branch2b', 'res4d_branch2c', 'res4e_branch2a', 'res4e_branch2b', 'res4e_branch2c', 'res4f_branch2a', 'res4f_branch2b', 'res4f_branch2c', 'res5a_branch2a', 'res5a_branch2b', 'res5a_branch2c', 'res5a_branch1', 'res5b_branch2a', 'res5b_branch2b', 'res5b_branch2c', 'res5c_branch2a', 'res5c_branch2b', 'res5c_branch2c', 'fc1000']
FLOPS: [498486016, 25890816, 231411712, 103563264, 103563264, 102961152, 231411712, 103563264, 102961152, 231411712, 103563264, 102860800, 231311360, 103161856, 411443200, 102860800, 231311360, 103161856, 102860800, 231311360, 103161856, 102860800, 231311360, 103161856, 102810624, 231261184, 102961152, 411242496, 102810624, 231261184, 102961152, 102810624, 231261184, 102961152, 102810624, 231261184, 102961152, 102810624, 231261184, 102961152, 102810624, 231261184, 102961152, 102785536, 231236096, 102860800, 411142144, 102785536, 231236096, 102860800, 102785536, 231236096, 102860800, 4097000]

Total FLOPS[GFLOPS]: 0.8758893288
