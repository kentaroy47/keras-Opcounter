# keras-Opcounter
calculate number of OPS in a Keras model!

## still under development! :)
Numbers are still buggy, plz wait till I fix this.

Right now, supports conv2d and dense only.

Doesn't fully count the activations yet.

We count 1 MAC as 2 FLOPS and 1 ADD as 1 FLOPS.
# Usage

## Count the ops in VGG16:

```
python vgg.py
```

```
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0
_________________________________________________________________
layer: block1_conv1 (None, 224, 224, 3)  MegaFLOPS: 173.40832  MegaWeights: 0.001792
layer: block1_conv2 (None, 224, 224, 64)  MegaFLOPS: 3699.376192  MegaWeights: 0.036928
layer: block2_conv1 (None, 112, 112, 64)  MegaFLOPS: 1849.688192  MegaWeights: 0.073856
layer: block2_conv2 (None, 112, 112, 128)  MegaFLOPS: 3699.376256  MegaWeights: 0.147584
layer: block3_conv1 (None, 56, 56, 128)  MegaFLOPS: 1849.68832  MegaWeights: 0.295168
layer: block3_conv2 (None, 56, 56, 256)  MegaFLOPS: 3699.376384  MegaWeights: 0.59008
layer: block3_conv3 (None, 56, 56, 256)  MegaFLOPS: 3699.376384  MegaWeights: 0.59008
layer: block4_conv1 (None, 28, 28, 256)  MegaFLOPS: 1849.688576  MegaWeights: 1.18016
layer: block4_conv2 (None, 28, 28, 512)  MegaFLOPS: 3699.37664  MegaWeights: 2.359808
layer: block4_conv3 (None, 28, 28, 512)  MegaFLOPS: 3699.37664  MegaWeights: 2.359808
layer: block5_conv1 (None, 14, 14, 512)  MegaFLOPS: 924.844544  MegaWeights: 2.359808
layer: block5_conv2 (None, 14, 14, 512)  MegaFLOPS: 924.844544  MegaWeights: 2.359808
layer: block5_conv3 (None, 14, 14, 512)  MegaFLOPS: 924.844544  MegaWeights: 2.359808
layer: fc1 (None, 25088)  MegaFLOPS: 205.524992  MegaWeights: 102.764544
layer: fc2 (None, 4096)  MegaFLOPS: 33.558528  MegaWeights: 16.781312
Total FLOPS[GFLOPS]: 30.932349056
```

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

Total FLOPS[GFLOPS]: 8.758893288
