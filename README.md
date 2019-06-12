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
layer: block1_conv1  MegaFLOPS: 176.61952
layer: block1_conv2  MegaFLOPS: 3702.587392
layer: block2_conv1  MegaFLOPS: 1851.293696
layer: block2_conv2  MegaFLOPS: 3700.98176
layer: block3_conv1  MegaFLOPS: 1850.49088
layer: block3_conv2  MegaFLOPS: 3700.178944
layer: block3_conv3  MegaFLOPS: 3700.178944
layer: block4_conv1  MegaFLOPS: 1850.089472
layer: block4_conv2  MegaFLOPS: 3699.777536
layer: block4_conv3  MegaFLOPS: 3699.777536
layer: block5_conv1  MegaFLOPS: 924.944384
layer: block5_conv2  MegaFLOPS: 924.944384
layer: block5_conv3  MegaFLOPS: 924.944384
layer: fc1  MegaFLOPS: 205.524992
layer: fc2  MegaFLOPS: 33.558528
Total FLOPS[GFLOPS]: 30.945892352
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
