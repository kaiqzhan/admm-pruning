import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def forward_all(self, x):
        res = dict()
        res['conv1'] = F.relu(self.conv1(x))
        res['pool1'] = F.max_pool2d(res['conv1'], 2, 2)
        res['conv2'] = F.relu(self.conv2(res['pool1']))
        res['pool2'] = F.max_pool2d(res['conv2'], 2, 2)
        x = res['pool2'].view(-1, 4*4*50)
        res['fc1'] = F.relu(self.fc1(x))
        res['fc2'] = self.fc2(res['fc1'])
        res['pred'] = F.log_softmax(res['fc2'], dim=1)
        return res
