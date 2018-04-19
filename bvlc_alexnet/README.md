## Model
The model definitions comes from BVLC/Caffe -> models -> bvlc_alexnet. You can run and test it with the caffe original prototxt files.

## Download links
0. bvlc_alexnet (focus on weight reduction)
  - [bvlc_alexnet_18x_total.caffemodel] (https://goo.gl/LbG21j)
  - [bvlc_alexnet_20x_total.caffemodel] (https://goo.gl/dcrdC6)
0. bvlc_alexnet (focus on conv reduction)
  - [bvlc_alexnet_13_4x_conv.caffemodel] (https://goo.gl/yiXQ2Y)

## Data Preprocessing
We use 'caffe/examples/imagenet/create_imagenet.sh' to convert the ImageNet dataset to lmdb files. We set 'RESIZE=true' in the script to resize the images to 256x256.
