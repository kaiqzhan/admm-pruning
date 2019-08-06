from __future__ import print_function
import argparse
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from pathlib import Path

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

    def prune(self, x):
        y = self.forward_all(x)
        order = ['conv1', 'conv2', 'fc1', 'fc2']
        full_order = ['conv1', 'pool1', 'conv2', 'pool2', 'fc1', 'fc2']
        for weight_name, W in self.named_parameters():
            if not weight_name.endswith('.weight'):
                continue
            name = weight_name[:-7] # remove .weight
            i = [i for i, s in enumerate(order) if name == s][0]
            mask = torch.norm(W.view(W.shape[0], -1), dim=1) == 0
            l_y = y[name]
            #print(name, l_y[:, mask].detach().cpu().numpy(), l_y.shape)

def train(args, model, device, train_loader, optimizer, epoch, mask={}):
    model.train()
    bs = train_loader.batch_size
    n = len(train_loader.dataset)
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        for weight_name, (W, m) in mask.items():
            W.grad[m] = 0
        optimizer.step()
        if i % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{i*bs:5d}/{n} ({100.*i*bs/n:.0f}%)]\tLoss: {loss.item():.6f}')

def project(W, p):
    Z = W.clone().detach()
    Z = Z.view(-1)
    p = int(p * Z.numel()) // 100
    abs_Z = torch.abs(Z)
    v, _ = torch.kthvalue(abs_Z, p)
    mask = abs_Z <= v
    Z[mask] = 0
    Z = Z.view(W.shape)
    mask = mask.view(W.shape)
    return Z, mask

def project_column(W, p):
    N = W.shape[0]
    Z = W.clone().detach()
    Z = Z.view(N, -1)
    nz = torch.norm(Z, dim=0)
    p = int(p * nz.numel()) // 100
    v, _ = torch.kthvalue(nz, p)
    mask = (nz <= v).view(1, -1).repeat(N, 1)
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
    mask = (nz <= v).view(-1, 1).repeat(1, M)
    Z[mask] = 0
    Z = Z.view(W.shape)
    mask = mask.view(W.shape)
    return Z, mask

def admm_loss(model, aux):
    loss = 0
    for weight_name, (W, Z, U, _) in aux.items():
        loss += F.mse_loss(W+U, Z, reduction='sum')
    return loss

def update_aux(model, aux, iteration):
    with torch.no_grad():
        for weight_name, (W, Z, U, project_fun) in aux.items():
            Z, _ = project_fun(W + U)
            diff = W - Z
            U += diff
            aux[weight_name] = (W, Z, U, project_fun)
            print(f'{iteration}th iteration: {weight_name} gap {torch.norm(diff).item()}')

def train_admm(args, model, device, train_loader, optimizer, epoch, aux, rho=1e-4):
    n = len(train_loader.dataset)
    model.train()
    bs = train_loader.batch_size
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss1 = F.nll_loss(output, target)
        loss2 = rho * admm_loss(model, aux)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        if (i+1) % args.log_interval == 0:
            print(f'Train Epoch: {epoch:3} [{(i+1)*bs:5d}/{n} ({100.*(i+1)*bs/n:.0f}%)]\
                    \tLoss: {loss.item():.6f}={loss1.item():.6f}+{loss2.item():.6f}')

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    n = len(test_loader.dataset)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= n

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{n} ({100.*correct/n:.0f}%)\n')

def log_model(model, filename):
    if not Path(filename).is_file():
        torch.save(model.state_dict(), filename)
    model.load_state_dict(torch.load(filename))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
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

    kwargs = {'num_workers': 7, 'pin_memory': True} if use_cuda else {}
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
    #optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5, amsgrad=True)

    #for epoch in range(1, args.epochs + 1):
    #    train(args, model, device, train_loader, optimizer, epoch)
    #    test(args, model, device, test_loader)

    log_model(model, 'mnist_cnn.pt')

    #p_array = [80, 92, 99.1, 93]
    p_array = [40, 60, 95]
    '''
    map from weight name to tuple (W, Z, U, project_fun)
    '''
    aux = {}
    for weight_name, W in model.named_parameters():
        if not weight_name.endswith('weight'):
            continue
        if weight_name == 'fc2.weight':
            continue
        project_fun = partial(project_filter, p=p_array[len(aux)])
        aux[weight_name] = (
                W,                                        # W
                project_fun(W)[0],                        # Z
                torch.zeros_like(W, requires_grad=False), # U
                project_fun,                              # project_fun
                )

    #j = 5
    #k = 30
    #optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5, amsgrad=True)
    #for i in range(k):
    #    for epoch in range(i*j, (i+1)*j):
    #        train_admm(args, model, device, train_loader, optimizer, epoch+1, aux)
    #    test(args, model, device, test_loader)
    #    update_aux(model, aux, i+1)
    #    for weight_name, (W, _, U, _) in aux.items():
    #        print(f'{weight_name} U norm {torch.norm(U).item():.6f}')

    log_model(model, 'mnist_cnn_admm.pt')

    # prepare mask
    with torch.no_grad():
        mask = {}
        for weight_name, (W, _, _, project_fun) in aux.items():
            _, m = project_fun(W)
            W[m] = 0
            mask[weight_name] = (W, m)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5, amsgrad=True)
    for weight_name, (W, _) in mask.items():
        c = (W != 0).sum()
        print(f'{weight_name}: {c}/{W.numel()} ({100.*float(c)/W.numel():.2f}%) non-zero values')

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, mask)
        test(args, model, device, test_loader)

    log_model(model, "mnist_cnn_admm_tuned.pt")

    x, _ = next(iter(train_loader))
    x = x.to(device)
    model.prune(x)

    for weight_name, (W, _) in mask.items():
        c = (W != 0).sum()
        print(f'{weight_name}: {c}/{W.numel()} ({100.*float(c)/W.numel():.2f}%) non-zero values')


if __name__ == '__main__':
    main()
