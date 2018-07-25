# admm-pruning
Prune DNN using Alternating Direction Method of Multipliers (ADMM)

## Our paper
“A Systematic DNN Weight Pruning Framework using Alternating Direction Method of Multipliers” ([https://arxiv.org/abs/1804.03294](https://arxiv.org/abs/1804.03294))

## Citation
If you use these models in your research, please cite:
```
@article{zhang2018systematic,
  title={A Systematic DNN Weight Pruning Framework using Alternating Direction Method of Multipliers},
  author={Zhang, Tianyun and Ye, Shaokai and Zhang, Kaiqi and Tang, Jian and Wen, Wujie and Fardad, Makan and Wang, Yanzhi},
  journal={arXiv preprint arXiv:1804.03294},
  year={2018}
}
```

## Models
1. lenet-5
  - see `tensorflow-mnist-model` in this repository
2. bvlc_alexnet (focus on weight reduction)
  - [bvlc_alexnet_21x_total.caffemodel] (http://bit.ly/2JVwKFD)
3. bvlc_alexnet (focus on conv reduction)
  - [bvlc_alexnet_13_4x_conv.caffemodel] (http://bit.ly/2uNotPv)

## Results
1. lenet-5 (top1 accuracy: 99.2%)

| Layer | Weights | Weights after prune | Weights after prune % |
| ----- | ------- | ------------------- | --------------------- |
| conv1 | 0.5K    | 0.1K                | 20%                   |
| conv2 | 25K     | 2K                  | 8%                    |
| fc1   | 400K    | 3.6K                | 0.9%                  |
| fc2   | 5K      | 0.35K               | 7%                    |
| Total | 430.5K  | 6.05K               | 1.4%                  |

2. bvlc_alexnet (top5 accuracy: 80.2%, 40 iterations of ADMM)

| Layer | Weights | Weights after prune | Weights after prune % |
| ----- | ------- | ------------------- | --------------------- |
| conv1 | 34.8K   | 28.19K              | 81%                   |
| conv2 | 307.2K  | 61.44K              | 20%                   |
| conv3 | 884.7K  | 168.09K             | 19%                   |
| conv4 | 663.5K  | 132.7K              | 20%                   |
| conv5 | 442.4K  | 88.48K              | 20%                   |
| fc1   | 37.7M   | 1.06M               | 2.8%                  |
| fc2   | 16.8M   | 0.99M               | 5.9%                  |
| fc3   | 4.1M    | 0.38M               | 9.3%                  |
| Total | 60.9M   | 2.9M                | 4.76%                 |
