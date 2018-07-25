## How to train

Two steps are required to prune a neural network. Suppose we have a pretrained model "bvlc_alexnet.caffemodel".

Step 1: admm (suppress target weights)
```
./build/tools/caffe train --solver=models/bvlc_alexnet/solver_admm.prototxt --weights=/path/to/bvlc_alexnet.caffemodel
```

Step 2: retrain (set target weights to 0 and retrain)
```
./build/tools/caffe train --solver=models/bvlc_alexnet/solver_retrain.prototxt --weights=models/bvlc_alexnet/caffe_alexnet_train_admm_iter_2400000.caffemodel
```
Suppose `caffe_alexnet_train_admm_iter_2400000.caffemodel` is the trained caffemodel from step 1.

## Explanation of parameters
### solver prototxt

- **lr_policy**: "admm" for admm step; your prefered policy for retrain step
- **gamma**: drop the learning rate by a factor of this value
- **stepvalue**: drop the learning rate by gamma when meet this stepvalue in each admm iteration
- **admm_iter**: Caffe training iterations for each admm iteration
- **max_iter**: Total training iterations (total admm iterations = max_iter / admm_iter)
- **pruning_phase**: "admm" for admm step and "retrain" for retrain step

### model prototxt

- **prune_ratio**: the percent of weight parameters to prune for each layer
- **rho**: the rho parameter for admm pruning (detailed in our paper)