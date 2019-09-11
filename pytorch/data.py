from torchvision import datasets, transforms
import torch

def loaders(use_cuda, train_batch_size, test_batch_size):
    kwargs = {'num_workers': 7, 'pin_memory': True} if use_cuda else {}
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,transform=tfm),
        batch_size=train_batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=tfm),
        batch_size=test_batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader
