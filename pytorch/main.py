from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path

from net import Net
from options import options
from data import loaders
from admm_utils import init_aux, init_mask, admm_loss, update_aux

def train(model,
        train_loader,
        optimizer,
        device,
        epoch,
        aux=None,   # auxiliary variables
        mask={},    # mask of weights to be fixed
        rho=1e-4,   # parameter of the augmented Langrangian
        log_interval=100):

    model.train()
    bs = train_loader.batch_size
    n = len(train_loader.dataset)
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)

        loss1 = F.nll_loss(output, target)
        loss2 = torch.Tensor([0])[0] if aux is None else rho * admm_loss(aux)
        loss = loss1 + loss2

        loss.backward()
        for weight_name, (W, m) in mask.items(): # used for finetuning
            W.grad[m] = 0
        optimizer.step()

        j = i+1
        if j % log_interval == 0:
            print(f'Train Epoch: {epoch:3} [{j*bs:5d}/{n} ({100.*j*bs/n:.0f}%)]\
                    \tLoss: {loss.item():.6f}={loss1.item():.6f}+{loss2.item():.6f}')
    j = i+1
    print(f'Train Epoch: {epoch:3} [{n:5d}/{n} ({100.*j*bs/n:.0f}%)]\
            \tLoss: {loss.item():.6f}={loss1.item():.6f}+{loss2.item():.6f}')


def test(model, test_loader, device):
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

    print(f'\nTest set: Average loss: {test_loss/n:.4f}, Accuracy: {correct}/{n} ({100.*correct/n:.0f}%)\n')

def log_model(model, filename):
    filepath = Path('models')/filename
    torch.save(model.state_dict(), str(filepath))

def print_weight_statistics(mask):
    for weight_name, (W, _) in mask.items():
        c = (W != 0).sum()
        print(f'{weight_name}: {c}/{W.numel()} ({100.*float(c)/W.numel():.2f}%) non-zero values')

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch ADMM pruning Example')
    options(parser)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = Net().to(device)

    train_loader, test_loader = loaders(use_cuda, args.batch_size, args.test_batch_size)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5, amsgrad=True)

    # initial training
    for epoch in range(args.epochs):
        train(model, train_loader, optimizer, device, epoch+1)
        test(model, test_loader, device)

    log_model(model, 'mnist_cnn.pth')

    # pruning
    p_array = [40, 60, 95]
    aux = init_aux(model, p_array)

    j = 5
    k = 30
    for i in range(k):
        for epoch in range(i*j, (i+1)*j):
            train(model, train_loader, optimizer, device, epoch+1, aux=aux, log_interval=int(1e6))
        test(model, test_loader, device)
        update_aux(aux, i+1)
        for weight_name, (W, _, U, _) in aux.items():
            print(f'{weight_name} U norm {torch.norm(U).item():.6f}')

    log_model(model, 'mnist_cnn_admm.pth')

    # fine-tuning
    mask = init_mask(aux)

    # Important to reinitialize optimizer to reset momentum
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5, amsgrad=True)
    print_weight_statistics(mask)

    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, optimizer, device, epoch, mask=mask)
        test(model, test_loader, device)

    log_model(model, "mnist_cnn_admm_tuned.pth")

    print_weight_statistics(mask)

if __name__ == '__main__':
    main()
