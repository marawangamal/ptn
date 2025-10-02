import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader


def load_mnist(data_dir="./data", scale=None):
    transform = transforms.Compose([transforms.ToTensor(), lambda x: (x > 0.5).long()])
    if scale is not None:
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (scale, scale)
                ),  # rescale from 28x28 -> (scale, scale)
                transforms.ToTensor(),
                lambda x: (x > 0.5).long(),  # binarize
            ]
        )
    train_set = torchvision.datasets.MNIST(
        data_dir, train=True, transform=transform, download=True
    )
    val_set = torchvision.datasets.MNIST(
        data_dir, train=False, transform=transform, download=True
    )
    return train_set, val_set


def dataset_to_numpy(dataset):
    loader = DataLoader(dataset, batch_size=len(dataset))
    X, _ = next(iter(loader))  # (N, 1, 28, 28)
    return X.view(len(dataset), -1).numpy()  # reshape to (N, 28*28)


if __name__ == "__main__":
    train_set, test_set = load_mnist(scale=28)
    X_train = dataset_to_numpy(train_set)
    X_test = dataset_to_numpy(test_set)

    np.save("data/mnist/train.npy", X_train)
    np.save("data/mnist/test.npy", X_test)
    print("Train shape:", X_train.shape)  # (60000, 784)
    print("Test shape:", X_test.shape)  # (10000, 784)
