from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np


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

def train(args, model, device, train_loader, optimizer, epoch, mask={}):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        for weight_name, value in model.named_parameters():
            if not weight_name in mask:
                continue
            value.grad[mask[weight_name]] = 0
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def project(W, p):
    per = np.percentile(abs(W), p)
    print("percentile {}".format(per))
    mask = abs(W) < per
    W[mask] = 0
    return W, mask

def project_column(W, p):
    N = W.shape[0]
    Z = W.clone().detach()
    Z = Z.view(N, -1)
    nz = torch.norm(Z, dim=0)
    p = int(p * nz.numel()) // 100
    v, _ = torch.kthvalue(nz, p)
    mask = (nz < v).view(1, -1).repeat(N, 1)
    Z[mask] = 0
    Z = Z.view(W.shape)
    mask = mask.view(W.shape)
    return Z, mask

def project_filter(W, p):
    N = W.shape[0]
    Z = W.clone().detach()
    Z = Z.view(N, -1)
    M = Z.shape[1]
    nz = torch.norm(Z, dim=1)
    p = int(p * nz.numel()) // 100
    v, _ = torch.kthvalue(nz, p)
    mask = (nz < v).view(-1, 1).repeat(1, M)
    Z[mask] = 0
    Z = Z.view(W.shape)
    mask = mask.view(W.shape)
    return Z, mask

def admm_loss(model, aux):
    loss = 0
    for weight_name, value in model.named_parameters():
        if not weight_name in aux:
            continue
        Z, U = aux[weight_name]
        loss += torch.norm(value - torch.tensor(Z - U, device=value.device)) ** 2
    return loss

def update_aux(model, aux, prune_factor, iteration, rho=1e-2):
    with torch.no_grad():
        for weight_name, value in model.named_parameters():
            if not weight_name in aux:
                continue
            Z, U = aux[weight_name]
            value = value.cpu().detach().numpy()
            diff_old = value - Z
            Z, _ = project(value + U, prune_factor[weight_name])
            diff = value - Z
            U += rho * diff
            aux[weight_name] = (Z, U)
            print('{}th iteration: {} gap {}'.format(iteration, weight_name, np.linalg.norm(diff)))

def train_admm(args, model, device, train_loader, optimizer, epoch, aux, prune_factor, rho=1e-4):
    n = len(train_loader)
    k = len(str(n))
    t = 'Train Epoch: {} [{:' + str(k+2) + '}/{} ({:.0f}%)]\tLoss: {:.6f}={:.6f}+{:.6f}\tGap: {:.6f}'
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss1 = F.nll_loss(output, target)
        loss2 = rho * admm_loss(model, aux)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        if (batch_idx+1) % args.log_interval == 0:
            with torch.no_grad():
                diff = 0
                for weight_name, value in model.named_parameters():
                    if not weight_name in aux:
                        continue
                    Z, _ = aux[weight_name]
                    d = (value.detach().cpu().numpy() - Z).reshape(-1)
                    diff += np.inner(d, d)
            print(t.format(
                epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
                100. * (batch_idx+1) / len(train_loader), loss.item(), loss1.item(), loss2.item(), diff))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--admm-tune-interval', type=int, default=100, metavar='N',
                        help='how many batches use to tune model before pruning')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Net().to(device)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), weight_decay=5e-5)

    #for epoch in range(1, args.epochs + 1):
    #    train(args, model, device, train_loader, optimizer, epoch)
    #    test(args, model, device, test_loader)

    model.load_state_dict(torch.load("mnist_cnn.pt"))

    #test(args, model, device, test_loader)

    p_array = [80, 92, 99.1, 93]
    aux = {}
    prune_factor = {}
    for weight_name, value in model.named_parameters():
        if not weight_name.endswith('weight'):
            continue
        prune_factor[weight_name] = p_array[len(prune_factor)]
        value = value.cpu().detach().numpy()
        aux[weight_name] = (project(value, prune_factor[weight_name])[0],
                np.zeros_like(value))

    #j = 10
    #for i in range(30):
    #    for epoch in range(i*j, (i+1)*j):
    #        train_admm(args, model, device, train_loader, optimizer, epoch+1, aux, prune_factor)
    #    test(args, model, device, test_loader)
    #    update_aux(model, aux, prune_factor, i+1, 1)

    #torch.save(model.state_dict(),"mnist_cnn_admm.pt")

    model.load_state_dict(torch.load("mnist_cnn_admm.pt"))

    # prepare mask
    with torch.no_grad():
        mask = {}
        for weight_name, value in model.named_parameters():
            if not weight_name in prune_factor:
                continue
            _, mask[weight_name] = project(value.detach().cpu().numpy(), prune_factor[weight_name])
            mask[weight_name] = torch.tensor(mask[weight_name], device=device)
            value[mask[weight_name]] = 0

    for weight_name, value in model.named_parameters():
        if not weight_name in prune_factor:
            continue
        c = (value != 0).sum()
        print('{}: {}/{} ({:.2f}%) non-zero values'.format(weight_name, c, value.numel(), 100.*float(c)/value.numel()))

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, mask)
        test(args, model, device, test_loader)

    for weight_name, value in model.named_parameters():
        if not weight_name in prune_factor:
            continue
        c = (value != 0).sum()
        print('{}: {}/{} ({:.2f}%) non-zero values'.format(weight_name, c, value.numel(), 100.*float(c)/value.numel()))

if __name__ == '__main__':
    main()
