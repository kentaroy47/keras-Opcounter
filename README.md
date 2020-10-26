# KerOp: Keras-Opcounter

## How to install
`pip install kerop`

- **Counts the number of OPS in your Keras model!**

- **Visualizes the OPS at each of your layer, to find the bottleneck.**

Supported layers: conv2d, fc, residual.

# Usage

```python
import kerop

# analyze FLOPS
layer_name, layer_flops, inshape, weights = kerop.profile(your_keras_model)

# visualize results
for name, flop, shape, weight in zip(layer_name, layer_flops, inshape, weights):
    print("layer:", name, shape, " MegaFLOPS:", flop/1e6, " MegaWeights:", weight/1e6)
```

### great thanks
@Lyken17

https://github.com/Lyken17/pytorch-OpCounter

## Example: Count the ops in VGG16:

```
python example/vgg.py
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
python example/resnet.py
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

## densenet
```
python example/densenet.py

layer: conv1/conv (None, 230, 230, 3)  MegaFLOPS: 236.027904  MegaWeights: 0.009408
layer: conv2_block1_1_conv (None, 56, 56, 64)  MegaFLOPS: 51.380224  MegaWeights: 0.008192
layer: conv2_block1_2_conv (None, 56, 56, 128)  MegaFLOPS: 231.211008  MegaWeights: 0.036864
layer: conv2_block2_1_conv (None, 56, 56, 96)  MegaFLOPS: 77.070336  MegaWeights: 0.012288
layer: conv2_block2_2_conv (None, 56, 56, 128)  MegaFLOPS: 231.211008  MegaWeights: 0.036864
layer: conv2_block3_1_conv (None, 56, 56, 128)  MegaFLOPS: 102.760448  MegaWeights: 0.016384
layer: conv2_block3_2_conv (None, 56, 56, 128)  MegaFLOPS: 231.211008  MegaWeights: 0.036864
layer: conv2_block4_1_conv (None, 56, 56, 160)  MegaFLOPS: 128.45056  MegaWeights: 0.02048
layer: conv2_block4_2_conv (None, 56, 56, 128)  MegaFLOPS: 231.211008  MegaWeights: 0.036864
layer: conv2_block5_1_conv (None, 56, 56, 192)  MegaFLOPS: 154.140672  MegaWeights: 0.024576
layer: conv2_block5_2_conv (None, 56, 56, 128)  MegaFLOPS: 231.211008  MegaWeights: 0.036864
layer: conv2_block6_1_conv (None, 56, 56, 224)  MegaFLOPS: 179.830784  MegaWeights: 0.028672
layer: conv2_block6_2_conv (None, 56, 56, 128)  MegaFLOPS: 231.211008  MegaWeights: 0.036864
layer: pool2_conv (None, 56, 56, 256)  MegaFLOPS: 205.520896  MegaWeights: 0.032768
layer: conv3_block1_1_conv (None, 28, 28, 128)  MegaFLOPS: 25.690112  MegaWeights: 0.016384
layer: conv3_block1_2_conv (None, 28, 28, 128)  MegaFLOPS: 57.802752  MegaWeights: 0.036864
layer: conv3_block2_1_conv (None, 28, 28, 160)  MegaFLOPS: 32.11264  MegaWeights: 0.02048
layer: conv3_block2_2_conv (None, 28, 28, 128)  MegaFLOPS: 57.802752  MegaWeights: 0.036864
layer: conv3_block3_1_conv (None, 28, 28, 192)  MegaFLOPS: 38.535168  MegaWeights: 0.024576
layer: conv3_block3_2_conv (None, 28, 28, 128)  MegaFLOPS: 57.802752  MegaWeights: 0.036864
layer: conv3_block4_1_conv (None, 28, 28, 224)  MegaFLOPS: 44.957696  MegaWeights: 0.028672
layer: conv3_block4_2_conv (None, 28, 28, 128)  MegaFLOPS: 57.802752  MegaWeights: 0.036864
layer: conv3_block5_1_conv (None, 28, 28, 256)  MegaFLOPS: 51.380224  MegaWeights: 0.032768
layer: conv3_block5_2_conv (None, 28, 28, 128)  MegaFLOPS: 57.802752  MegaWeights: 0.036864
layer: conv3_block6_1_conv (None, 28, 28, 288)  MegaFLOPS: 57.802752  MegaWeights: 0.036864
layer: conv3_block6_2_conv (None, 28, 28, 128)  MegaFLOPS: 57.802752  MegaWeights: 0.036864
layer: conv3_block7_1_conv (None, 28, 28, 320)  MegaFLOPS: 64.22528  MegaWeights: 0.04096
layer: conv3_block7_2_conv (None, 28, 28, 128)  MegaFLOPS: 57.802752  MegaWeights: 0.036864
layer: conv3_block8_1_conv (None, 28, 28, 352)  MegaFLOPS: 70.647808  MegaWeights: 0.045056
layer: conv3_block8_2_conv (None, 28, 28, 128)  MegaFLOPS: 57.802752  MegaWeights: 0.036864
layer: conv3_block9_1_conv (None, 28, 28, 384)  MegaFLOPS: 77.070336  MegaWeights: 0.049152
layer: conv3_block9_2_conv (None, 28, 28, 128)  MegaFLOPS: 57.802752  MegaWeights: 0.036864
layer: conv3_block10_1_conv (None, 28, 28, 416)  MegaFLOPS: 83.492864  MegaWeights: 0.053248
layer: conv3_block10_2_conv (None, 28, 28, 128)  MegaFLOPS: 57.802752  MegaWeights: 0.036864
layer: conv3_block11_1_conv (None, 28, 28, 448)  MegaFLOPS: 89.915392  MegaWeights: 0.057344
layer: conv3_block11_2_conv (None, 28, 28, 128)  MegaFLOPS: 57.802752  MegaWeights: 0.036864
layer: conv3_block12_1_conv (None, 28, 28, 480)  MegaFLOPS: 96.33792  MegaWeights: 0.06144
layer: conv3_block12_2_conv (None, 28, 28, 128)  MegaFLOPS: 57.802752  MegaWeights: 0.036864
layer: pool3_conv (None, 28, 28, 512)  MegaFLOPS: 205.520896  MegaWeights: 0.131072
layer: conv4_block1_1_conv (None, 14, 14, 256)  MegaFLOPS: 12.845056  MegaWeights: 0.032768
layer: conv4_block1_2_conv (None, 14, 14, 128)  MegaFLOPS: 14.450688  MegaWeights: 0.036864
layer: conv4_block2_1_conv (None, 14, 14, 288)  MegaFLOPS: 14.450688  MegaWeights: 0.036864
layer: conv4_block2_2_conv (None, 14, 14, 128)  MegaFLOPS: 14.450688  MegaWeights: 0.036864
layer: conv4_block3_1_conv (None, 14, 14, 320)  MegaFLOPS: 16.05632  MegaWeights: 0.04096
layer: conv4_block3_2_conv (None, 14, 14, 128)  MegaFLOPS: 14.450688  MegaWeights: 0.036864
layer: conv4_block4_1_conv (None, 14, 14, 352)  MegaFLOPS: 17.661952  MegaWeights: 0.045056
layer: conv4_block4_2_conv (None, 14, 14, 128)  MegaFLOPS: 14.450688  MegaWeights: 0.036864
layer: conv4_block5_1_conv (None, 14, 14, 384)  MegaFLOPS: 19.267584  MegaWeights: 0.049152
layer: conv4_block5_2_conv (None, 14, 14, 128)  MegaFLOPS: 14.450688  MegaWeights: 0.036864
layer: conv4_block6_1_conv (None, 14, 14, 416)  MegaFLOPS: 20.873216  MegaWeights: 0.053248
layer: conv4_block6_2_conv (None, 14, 14, 128)  MegaFLOPS: 14.450688  MegaWeights: 0.036864
layer: conv4_block7_1_conv (None, 14, 14, 448)  MegaFLOPS: 22.478848  MegaWeights: 0.057344
layer: conv4_block7_2_conv (None, 14, 14, 128)  MegaFLOPS: 14.450688  MegaWeights: 0.036864
layer: conv4_block8_1_conv (None, 14, 14, 480)  MegaFLOPS: 24.08448  MegaWeights: 0.06144
layer: conv4_block8_2_conv (None, 14, 14, 128)  MegaFLOPS: 14.450688  MegaWeights: 0.036864
layer: conv4_block9_1_conv (None, 14, 14, 512)  MegaFLOPS: 25.690112  MegaWeights: 0.065536
layer: conv4_block9_2_conv (None, 14, 14, 128)  MegaFLOPS: 14.450688  MegaWeights: 0.036864
layer: conv4_block10_1_conv (None, 14, 14, 544)  MegaFLOPS: 27.295744  MegaWeights: 0.069632
layer: conv4_block10_2_conv (None, 14, 14, 128)  MegaFLOPS: 14.450688  MegaWeights: 0.036864
layer: conv4_block11_1_conv (None, 14, 14, 576)  MegaFLOPS: 28.901376  MegaWeights: 0.073728
layer: conv4_block11_2_conv (None, 14, 14, 128)  MegaFLOPS: 14.450688  MegaWeights: 0.036864
layer: conv4_block12_1_conv (None, 14, 14, 608)  MegaFLOPS: 30.507008  MegaWeights: 0.077824
layer: conv4_block12_2_conv (None, 14, 14, 128)  MegaFLOPS: 14.450688  MegaWeights: 0.036864
layer: conv4_block13_1_conv (None, 14, 14, 640)  MegaFLOPS: 32.11264  MegaWeights: 0.08192
layer: conv4_block13_2_conv (None, 14, 14, 128)  MegaFLOPS: 14.450688  MegaWeights: 0.036864
layer: conv4_block14_1_conv (None, 14, 14, 672)  MegaFLOPS: 33.718272  MegaWeights: 0.086016
layer: conv4_block14_2_conv (None, 14, 14, 128)  MegaFLOPS: 14.450688  MegaWeights: 0.036864
layer: conv4_block15_1_conv (None, 14, 14, 704)  MegaFLOPS: 35.323904  MegaWeights: 0.090112
layer: conv4_block15_2_conv (None, 14, 14, 128)  MegaFLOPS: 14.450688  MegaWeights: 0.036864
layer: conv4_block16_1_conv (None, 14, 14, 736)  MegaFLOPS: 36.929536  MegaWeights: 0.094208
layer: conv4_block16_2_conv (None, 14, 14, 128)  MegaFLOPS: 14.450688  MegaWeights: 0.036864
layer: conv4_block17_1_conv (None, 14, 14, 768)  MegaFLOPS: 38.535168  MegaWeights: 0.098304
layer: conv4_block17_2_conv (None, 14, 14, 128)  MegaFLOPS: 14.450688  MegaWeights: 0.036864
layer: conv4_block18_1_conv (None, 14, 14, 800)  MegaFLOPS: 40.1408  MegaWeights: 0.1024
layer: conv4_block18_2_conv (None, 14, 14, 128)  MegaFLOPS: 14.450688  MegaWeights: 0.036864
layer: conv4_block19_1_conv (None, 14, 14, 832)  MegaFLOPS: 41.746432  MegaWeights: 0.106496
layer: conv4_block19_2_conv (None, 14, 14, 128)  MegaFLOPS: 14.450688  MegaWeights: 0.036864
layer: conv4_block20_1_conv (None, 14, 14, 864)  MegaFLOPS: 43.352064  MegaWeights: 0.110592
layer: conv4_block20_2_conv (None, 14, 14, 128)  MegaFLOPS: 14.450688  MegaWeights: 0.036864
layer: conv4_block21_1_conv (None, 14, 14, 896)  MegaFLOPS: 44.957696  MegaWeights: 0.114688
layer: conv4_block21_2_conv (None, 14, 14, 128)  MegaFLOPS: 14.450688  MegaWeights: 0.036864
layer: conv4_block22_1_conv (None, 14, 14, 928)  MegaFLOPS: 46.563328  MegaWeights: 0.118784
layer: conv4_block22_2_conv (None, 14, 14, 128)  MegaFLOPS: 14.450688  MegaWeights: 0.036864
layer: conv4_block23_1_conv (None, 14, 14, 960)  MegaFLOPS: 48.16896  MegaWeights: 0.12288
layer: conv4_block23_2_conv (None, 14, 14, 128)  MegaFLOPS: 14.450688  MegaWeights: 0.036864
layer: conv4_block24_1_conv (None, 14, 14, 992)  MegaFLOPS: 49.774592  MegaWeights: 0.126976
layer: conv4_block24_2_conv (None, 14, 14, 128)  MegaFLOPS: 14.450688  MegaWeights: 0.036864
layer: conv4_block25_1_conv (None, 14, 14, 1024)  MegaFLOPS: 51.380224  MegaWeights: 0.131072
layer: conv4_block25_2_conv (None, 14, 14, 128)  MegaFLOPS: 14.450688  MegaWeights: 0.036864
layer: conv4_block26_1_conv (None, 14, 14, 1056)  MegaFLOPS: 52.985856  MegaWeights: 0.135168
layer: conv4_block26_2_conv (None, 14, 14, 128)  MegaFLOPS: 14.450688  MegaWeights: 0.036864
layer: conv4_block27_1_conv (None, 14, 14, 1088)  MegaFLOPS: 54.591488  MegaWeights: 0.139264
layer: conv4_block27_2_conv (None, 14, 14, 128)  MegaFLOPS: 14.450688  MegaWeights: 0.036864
layer: conv4_block28_1_conv (None, 14, 14, 1120)  MegaFLOPS: 56.19712  MegaWeights: 0.14336
layer: conv4_block28_2_conv (None, 14, 14, 128)  MegaFLOPS: 14.450688  MegaWeights: 0.036864
layer: conv4_block29_1_conv (None, 14, 14, 1152)  MegaFLOPS: 57.802752  MegaWeights: 0.147456
layer: conv4_block29_2_conv (None, 14, 14, 128)  MegaFLOPS: 14.450688  MegaWeights: 0.036864
layer: conv4_block30_1_conv (None, 14, 14, 1184)  MegaFLOPS: 59.408384  MegaWeights: 0.151552
layer: conv4_block30_2_conv (None, 14, 14, 128)  MegaFLOPS: 14.450688  MegaWeights: 0.036864
layer: conv4_block31_1_conv (None, 14, 14, 1216)  MegaFLOPS: 61.014016  MegaWeights: 0.155648
layer: conv4_block31_2_conv (None, 14, 14, 128)  MegaFLOPS: 14.450688  MegaWeights: 0.036864
layer: conv4_block32_1_conv (None, 14, 14, 1248)  MegaFLOPS: 62.619648  MegaWeights: 0.159744
layer: conv4_block32_2_conv (None, 14, 14, 128)  MegaFLOPS: 14.450688  MegaWeights: 0.036864
layer: pool4_conv (None, 14, 14, 1280)  MegaFLOPS: 321.1264  MegaWeights: 0.8192
layer: conv5_block1_1_conv (None, 7, 7, 640)  MegaFLOPS: 8.02816  MegaWeights: 0.08192
layer: conv5_block1_2_conv (None, 7, 7, 128)  MegaFLOPS: 3.612672  MegaWeights: 0.036864
layer: conv5_block2_1_conv (None, 7, 7, 672)  MegaFLOPS: 8.429568  MegaWeights: 0.086016
layer: conv5_block2_2_conv (None, 7, 7, 128)  MegaFLOPS: 3.612672  MegaWeights: 0.036864
layer: conv5_block3_1_conv (None, 7, 7, 704)  MegaFLOPS: 8.830976  MegaWeights: 0.090112
layer: conv5_block3_2_conv (None, 7, 7, 128)  MegaFLOPS: 3.612672  MegaWeights: 0.036864
layer: conv5_block4_1_conv (None, 7, 7, 736)  MegaFLOPS: 9.232384  MegaWeights: 0.094208
layer: conv5_block4_2_conv (None, 7, 7, 128)  MegaFLOPS: 3.612672  MegaWeights: 0.036864
layer: conv5_block5_1_conv (None, 7, 7, 768)  MegaFLOPS: 9.633792  MegaWeights: 0.098304
layer: conv5_block5_2_conv (None, 7, 7, 128)  MegaFLOPS: 3.612672  MegaWeights: 0.036864
layer: conv5_block6_1_conv (None, 7, 7, 800)  MegaFLOPS: 10.0352  MegaWeights: 0.1024
layer: conv5_block6_2_conv (None, 7, 7, 128)  MegaFLOPS: 3.612672  MegaWeights: 0.036864
layer: conv5_block7_1_conv (None, 7, 7, 832)  MegaFLOPS: 10.436608  MegaWeights: 0.106496
layer: conv5_block7_2_conv (None, 7, 7, 128)  MegaFLOPS: 3.612672  MegaWeights: 0.036864
layer: conv5_block8_1_conv (None, 7, 7, 864)  MegaFLOPS: 10.838016  MegaWeights: 0.110592
layer: conv5_block8_2_conv (None, 7, 7, 128)  MegaFLOPS: 3.612672  MegaWeights: 0.036864
layer: conv5_block9_1_conv (None, 7, 7, 896)  MegaFLOPS: 11.239424  MegaWeights: 0.114688
layer: conv5_block9_2_conv (None, 7, 7, 128)  MegaFLOPS: 3.612672  MegaWeights: 0.036864
layer: conv5_block10_1_conv (None, 7, 7, 928)  MegaFLOPS: 11.640832  MegaWeights: 0.118784
layer: conv5_block10_2_conv (None, 7, 7, 128)  MegaFLOPS: 3.612672  MegaWeights: 0.036864
layer: conv5_block11_1_conv (None, 7, 7, 960)  MegaFLOPS: 12.04224  MegaWeights: 0.12288
layer: conv5_block11_2_conv (None, 7, 7, 128)  MegaFLOPS: 3.612672  MegaWeights: 0.036864
layer: conv5_block12_1_conv (None, 7, 7, 992)  MegaFLOPS: 12.443648  MegaWeights: 0.126976
layer: conv5_block12_2_conv (None, 7, 7, 128)  MegaFLOPS: 3.612672  MegaWeights: 0.036864
layer: conv5_block13_1_conv (None, 7, 7, 1024)  MegaFLOPS: 12.845056  MegaWeights: 0.131072
layer: conv5_block13_2_conv (None, 7, 7, 128)  MegaFLOPS: 3.612672  MegaWeights: 0.036864
layer: conv5_block14_1_conv (None, 7, 7, 1056)  MegaFLOPS: 13.246464  MegaWeights: 0.135168
layer: conv5_block14_2_conv (None, 7, 7, 128)  MegaFLOPS: 3.612672  MegaWeights: 0.036864
layer: conv5_block15_1_conv (None, 7, 7, 1088)  MegaFLOPS: 13.647872  MegaWeights: 0.139264
layer: conv5_block15_2_conv (None, 7, 7, 128)  MegaFLOPS: 3.612672  MegaWeights: 0.036864
layer: conv5_block16_1_conv (None, 7, 7, 1120)  MegaFLOPS: 14.04928  MegaWeights: 0.14336
layer: conv5_block16_2_conv (None, 7, 7, 128)  MegaFLOPS: 3.612672  MegaWeights: 0.036864
layer: conv5_block17_1_conv (None, 7, 7, 1152)  MegaFLOPS: 14.450688  MegaWeights: 0.147456
layer: conv5_block17_2_conv (None, 7, 7, 128)  MegaFLOPS: 3.612672  MegaWeights: 0.036864
layer: conv5_block18_1_conv (None, 7, 7, 1184)  MegaFLOPS: 14.852096  MegaWeights: 0.151552
layer: conv5_block18_2_conv (None, 7, 7, 128)  MegaFLOPS: 3.612672  MegaWeights: 0.036864
layer: conv5_block19_1_conv (None, 7, 7, 1216)  MegaFLOPS: 15.253504  MegaWeights: 0.155648
layer: conv5_block19_2_conv (None, 7, 7, 128)  MegaFLOPS: 3.612672  MegaWeights: 0.036864
layer: conv5_block20_1_conv (None, 7, 7, 1248)  MegaFLOPS: 15.654912  MegaWeights: 0.159744
layer: conv5_block20_2_conv (None, 7, 7, 128)  MegaFLOPS: 3.612672  MegaWeights: 0.036864
layer: conv5_block21_1_conv (None, 7, 7, 1280)  MegaFLOPS: 16.05632  MegaWeights: 0.16384
layer: conv5_block21_2_conv (None, 7, 7, 128)  MegaFLOPS: 3.612672  MegaWeights: 0.036864
layer: conv5_block22_1_conv (None, 7, 7, 1312)  MegaFLOPS: 16.457728  MegaWeights: 0.167936
layer: conv5_block22_2_conv (None, 7, 7, 128)  MegaFLOPS: 3.612672  MegaWeights: 0.036864
layer: conv5_block23_1_conv (None, 7, 7, 1344)  MegaFLOPS: 16.859136  MegaWeights: 0.172032
layer: conv5_block23_2_conv (None, 7, 7, 128)  MegaFLOPS: 3.612672  MegaWeights: 0.036864
layer: conv5_block24_1_conv (None, 7, 7, 1376)  MegaFLOPS: 17.260544  MegaWeights: 0.176128
layer: conv5_block24_2_conv (None, 7, 7, 128)  MegaFLOPS: 3.612672  MegaWeights: 0.036864
layer: conv5_block25_1_conv (None, 7, 7, 1408)  MegaFLOPS: 17.661952  MegaWeights: 0.180224
layer: conv5_block25_2_conv (None, 7, 7, 128)  MegaFLOPS: 3.612672  MegaWeights: 0.036864
layer: conv5_block26_1_conv (None, 7, 7, 1440)  MegaFLOPS: 18.06336  MegaWeights: 0.18432
layer: conv5_block26_2_conv (None, 7, 7, 128)  MegaFLOPS: 3.612672  MegaWeights: 0.036864
layer: conv5_block27_1_conv (None, 7, 7, 1472)  MegaFLOPS: 18.464768  MegaWeights: 0.188416
layer: conv5_block27_2_conv (None, 7, 7, 128)  MegaFLOPS: 3.612672  MegaWeights: 0.036864
layer: conv5_block28_1_conv (None, 7, 7, 1504)  MegaFLOPS: 18.866176  MegaWeights: 0.192512
layer: conv5_block28_2_conv (None, 7, 7, 128)  MegaFLOPS: 3.612672  MegaWeights: 0.036864
layer: conv5_block29_1_conv (None, 7, 7, 1536)  MegaFLOPS: 19.267584  MegaWeights: 0.196608
layer: conv5_block29_2_conv (None, 7, 7, 128)  MegaFLOPS: 3.612672  MegaWeights: 0.036864
layer: conv5_block30_1_conv (None, 7, 7, 1568)  MegaFLOPS: 19.668992  MegaWeights: 0.200704
layer: conv5_block30_2_conv (None, 7, 7, 128)  MegaFLOPS: 3.612672  MegaWeights: 0.036864
layer: conv5_block31_1_conv (None, 7, 7, 1600)  MegaFLOPS: 20.0704  MegaWeights: 0.2048
layer: conv5_block31_2_conv (None, 7, 7, 128)  MegaFLOPS: 3.612672  MegaWeights: 0.036864
layer: conv5_block32_1_conv (None, 7, 7, 1632)  MegaFLOPS: 20.471808  MegaWeights: 0.208896
layer: conv5_block32_2_conv (None, 7, 7, 128)  MegaFLOPS: 3.612672  MegaWeights: 0.036864
layer: fc1000 (None, 1664)  MegaFLOPS: 3.329  MegaWeights: 1.665
Total FLOPS[GFLOPS]: 6.719687656
```

## still under development! :)
Numbers are still buggy, plz wait till I fix this.

Right now, supports conv2d and dense only.

Doesn't fully count the activations yet.

We count 1 MAC as 2 FLOPS and 1 ADD as 1 FLOPS.
