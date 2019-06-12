# keras-Opcounter
calculate number of OPS in a Keras model!

# Usage

```
from util import profile
# analyze FLOPS
layer_name, layer_flops, inshape, weights = profile(your_keras_model)

# visualize results
for name, flop, shape, weight in zip(layer_name, layer_flops, inshape, weights):
    print("layer:", name, shape, " MegaFLOPS:", flop/1e6, " MegaWeights:", weight/1e6)
```

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

```
Total params: 25,636,712
Trainable params: 25,583,592
Non-trainable params: 53,120
__________________________________________________________________________________________________
layer: conv1 (None, 230, 230, 3)  MegaFLOPS: 497.683264  MegaWeights: 0.009472
layer: res2a_branch2a (None, 56, 56, 64)  MegaFLOPS: 25.690176  MegaWeights: 0.00416
layer: res2a_branch2b (None, 56, 56, 64)  MegaFLOPS: 231.211072  MegaWeights: 0.036928
layer: res2a_branch2c (None, 56, 56, 64)  MegaFLOPS: 102.760704  MegaWeights: 0.01664
layer: res2a_branch1 (None, 56, 56, 64)  MegaFLOPS: 102.760704  MegaWeights: 0.01664
layer: res2b_branch2a (None, 56, 56, 256)  MegaFLOPS: 102.760512  MegaWeights: 0.016448
layer: res2b_branch2b (None, 56, 56, 64)  MegaFLOPS: 231.211072  MegaWeights: 0.036928
layer: res2b_branch2c (None, 56, 56, 64)  MegaFLOPS: 102.760704  MegaWeights: 0.01664
layer: res2c_branch2a (None, 56, 56, 256)  MegaFLOPS: 102.760512  MegaWeights: 0.016448
layer: res2c_branch2b (None, 56, 56, 64)  MegaFLOPS: 231.211072  MegaWeights: 0.036928
layer: res2c_branch2c (None, 56, 56, 64)  MegaFLOPS: 102.760704  MegaWeights: 0.01664
layer: res3a_branch2a (None, 56, 56, 256)  MegaFLOPS: 102.760576  MegaWeights: 0.032896
layer: res3a_branch2b (None, 28, 28, 128)  MegaFLOPS: 231.211136  MegaWeights: 0.147584
layer: res3a_branch2c (None, 28, 28, 128)  MegaFLOPS: 102.76096  MegaWeights: 0.066048
layer: res3a_branch1 (None, 56, 56, 256)  MegaFLOPS: 411.042304  MegaWeights: 0.131584
layer: res3b_branch2a (None, 28, 28, 512)  MegaFLOPS: 102.760576  MegaWeights: 0.065664
layer: res3b_branch2b (None, 28, 28, 128)  MegaFLOPS: 231.211136  MegaWeights: 0.147584
layer: res3b_branch2c (None, 28, 28, 128)  MegaFLOPS: 102.76096  MegaWeights: 0.066048
layer: res3c_branch2a (None, 28, 28, 512)  MegaFLOPS: 102.760576  MegaWeights: 0.065664
layer: res3c_branch2b (None, 28, 28, 128)  MegaFLOPS: 231.211136  MegaWeights: 0.147584
layer: res3c_branch2c (None, 28, 28, 128)  MegaFLOPS: 102.76096  MegaWeights: 0.066048
layer: res3d_branch2a (None, 28, 28, 512)  MegaFLOPS: 102.760576  MegaWeights: 0.065664
layer: res3d_branch2b (None, 28, 28, 128)  MegaFLOPS: 231.211136  MegaWeights: 0.147584
layer: res3d_branch2c (None, 28, 28, 128)  MegaFLOPS: 102.76096  MegaWeights: 0.066048
layer: res4a_branch2a (None, 28, 28, 512)  MegaFLOPS: 102.760704  MegaWeights: 0.131328
layer: res4a_branch2b (None, 14, 14, 256)  MegaFLOPS: 231.211264  MegaWeights: 0.59008
layer: res4a_branch2c (None, 14, 14, 256)  MegaFLOPS: 102.761472  MegaWeights: 0.263168
layer: res4a_branch1 (None, 28, 28, 512)  MegaFLOPS: 411.042816  MegaWeights: 0.525312
layer: res4b_branch2a (None, 14, 14, 1024)  MegaFLOPS: 102.760704  MegaWeights: 0.2624
layer: res4b_branch2b (None, 14, 14, 256)  MegaFLOPS: 231.211264  MegaWeights: 0.59008
layer: res4b_branch2c (None, 14, 14, 256)  MegaFLOPS: 102.761472  MegaWeights: 0.263168
layer: res4c_branch2a (None, 14, 14, 1024)  MegaFLOPS: 102.760704  MegaWeights: 0.2624
layer: res4c_branch2b (None, 14, 14, 256)  MegaFLOPS: 231.211264  MegaWeights: 0.59008
layer: res4c_branch2c (None, 14, 14, 256)  MegaFLOPS: 102.761472  MegaWeights: 0.263168
layer: res4d_branch2a (None, 14, 14, 1024)  MegaFLOPS: 102.760704  MegaWeights: 0.2624
layer: res4d_branch2b (None, 14, 14, 256)  MegaFLOPS: 231.211264  MegaWeights: 0.59008
layer: res4d_branch2c (None, 14, 14, 256)  MegaFLOPS: 102.761472  MegaWeights: 0.263168
layer: res4e_branch2a (None, 14, 14, 1024)  MegaFLOPS: 102.760704  MegaWeights: 0.2624
layer: res4e_branch2b (None, 14, 14, 256)  MegaFLOPS: 231.211264  MegaWeights: 0.59008
layer: res4e_branch2c (None, 14, 14, 256)  MegaFLOPS: 102.761472  MegaWeights: 0.263168
layer: res4f_branch2a (None, 14, 14, 1024)  MegaFLOPS: 102.760704  MegaWeights: 0.2624
layer: res4f_branch2b (None, 14, 14, 256)  MegaFLOPS: 231.211264  MegaWeights: 0.59008
layer: res4f_branch2c (None, 14, 14, 256)  MegaFLOPS: 102.761472  MegaWeights: 0.263168
layer: res5a_branch2a (None, 14, 14, 1024)  MegaFLOPS: 102.76096  MegaWeights: 0.5248
layer: res5a_branch2b (None, 7, 7, 512)  MegaFLOPS: 231.21152  MegaWeights: 2.359808
layer: res5a_branch2c (None, 7, 7, 512)  MegaFLOPS: 102.762496  MegaWeights: 1.050624
layer: res5a_branch1 (None, 14, 14, 1024)  MegaFLOPS: 411.04384  MegaWeights: 2.0992
layer: res5b_branch2a (None, 7, 7, 2048)  MegaFLOPS: 102.76096  MegaWeights: 1.049088
layer: res5b_branch2b (None, 7, 7, 512)  MegaFLOPS: 231.21152  MegaWeights: 2.359808
layer: res5b_branch2c (None, 7, 7, 512)  MegaFLOPS: 102.762496  MegaWeights: 1.050624
layer: res5c_branch2a (None, 7, 7, 2048)  MegaFLOPS: 102.76096  MegaWeights: 1.049088
layer: res5c_branch2b (None, 7, 7, 512)  MegaFLOPS: 231.21152  MegaWeights: 2.359808
layer: res5c_branch2c (None, 7, 7, 512)  MegaFLOPS: 102.762496  MegaWeights: 1.050624
layer: fc1000 (None, 2048)  MegaFLOPS: 4.097  MegaWeights: 2.049
Total FLOPS[GFLOPS]: 8.748332712
```

## still under development! :)
Numbers are still buggy, plz wait till I fix this.

Right now, supports conv2d and dense only.

Doesn't fully count the activations yet.

We count 1 MAC as 2 FLOPS and 1 ADD as 1 FLOPS.