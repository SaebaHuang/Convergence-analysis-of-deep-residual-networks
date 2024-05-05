#enccoding=UTF-8
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def normalize_dataset(data):
    mean = data.mean(axis=(0, 1, 2)) / 255.0
    std = data.std(axis=(0, 1, 2)) / 255.0
    normalize = transforms.Normalize(mean=mean, std=std)
    print('Normalize -- mean: {},  std: {}'.format(mean, std))
    return normalize


def load_cifar10(dataset_path):
    import torch
    USE_GPU = True
    dtype = torch.float32
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('using device:', device)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_set = datasets.CIFAR10(root=dataset_path, train=True, download=False, transform=train_transform)
    test_set = datasets.CIFAR10(root=dataset_path, train=False, download=False, transform=test_transform)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=50000,
        shuffle=False
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=10000,
        shuffle=False
    )

    X_train, Y_train = next(iter(train_loader))
    X_test, Y_test = next(iter(test_loader))

    X_train = X_train.numpy()
    X_train = np.transpose(X_train, axes=(0, 2, 3, 1))
    Y_train = Y_train.numpy()

    X_test = X_test.numpy()
    X_test = np.transpose(X_test, axes=(0, 2, 3, 1))
    Y_test = Y_test.numpy()

    return X_train, Y_train, X_test, Y_test


if __name__ == '__main__':
    # test
    load_cifar10('./dataset/cifar10')




