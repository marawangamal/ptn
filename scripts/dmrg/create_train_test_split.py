import os
import certifi
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader

from urllib.request import urlretrieve, urlopen
from tqdm import tqdm

os.environ["SSL_CERT_FILE"] = certifi.where()


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


def load_ucla(
    batch_size=32,
    data_dir="./data",
    dataset="nltcs",
    max_samples=None,
):
    """Create MNIST data loaders with binary thresholding."""

    URI = "https://raw.githubusercontent.com/UCLA-StarAI/Density-Estimation-Datasets/refs/heads/master/datasets/"

    URLS = {
        "nltcs": {
            "train": URI + "nltcs/nltcs.train.data",
            "val": URI + "nltcs/nltcs.test.data",
        },
        "msnbc": {
            "train": URI + "msnbc/msnbc.train.data",
            "val": URI + "msnbc/msnbc.test.data",
        },
        "kdd": {
            "train": URI + "kdd/kdd.train.data",
            "val": URI + "kdd/kdd.test.data",
        },
        "plants": {
            "train": URI + "plants/plants.train.data",
            "val": URI + "plants/plants.test.data",
        },
        "baudio": {
            "train": URI + "baudio/baudio.train.data",
            "val": URI + "baudio/baudio.test.data",
        },
        "jester": {
            "train": URI + "jester/jester.train.data",
            "val": URI + "jester/jester.test.data",
        },
        "bnetflix": {
            "train": URI + "bnetflix/bnetflix.train.data",
            "val": URI + "bnetflix/bnetflix.test.data",
        },
        "accidents": {
            "train": URI + "accidents/accidents.train.data",
            "val": URI + "accidents/accidents.test.data",
        },
        "retail": {
            "train": URI + "tretail/tretail.train.data",
            "val": URI + "tretail/tretail.test.data",
        },
        "pumsb_star": {
            "train": URI + "pumsb_star/pumsb_star.train.data",
            "val": URI + "pumsb_star/pumsb_star.test.data",
        },
        "dna": {
            "train": URI + "dna/dna.train.data",
            "val": URI + "dna/dna.test.data",
        },
        "kosarek": {
            "train": URI + "kosarek/kosarek.train.data",
            "val": URI + "kosarek/kosarek.test.data",
        },
        "msweb": {
            "train": URI + "msweb/msweb.train.data",
            "val": URI + "msweb/msweb.test.data",
        },
        "book": {
            "train": URI + "book/book.train.data",
            "val": URI + "book/book.test.data",
        },
        "eachmovie": {
            "train": URI + "tmovie/tmovie.train.data",
            "val": URI + "tmovie/tmovie.test.data",
        },
        "webkb": {
            "train": URI + "webkb/webkb.train.data",
            "val": URI + "webkb/webkb.test.data",
        },
        "reuters_52": {
            "train": URI + "reuters_52/reuters_52.train.data",
            "val": URI + "reuters_52/reuters_52.test.data",
        },
        "c20ng": {
            "train": URI + "c20ng/c20ng.train.data",
            "val": URI + "c20ng/c20ng.test.data",
        },
        "bbc": {
            "train": URI + "bbc/bbc.train.data",
            "val": URI + "bbc/bbc.test.data",
        },
        "ad": {
            "train": URI + "ad/ad.train.data",
            "val": URI + "ad/ad.test.data",
        },
        "nips": {
            "train": URI + "nips/nips.train.data",
            "val": URI + "nips/nips.test.data",
        },
        "voting": {
            "train": URI + "voting/voting.train.data",
            "val": URI + "voting/voting.test.data",
        },
        "moviereview": {
            "train": URI + "moviereview/moviereview.train.data",
            "val": URI + "moviereview/moviereview.test.data",
        },
        "mushrooms": {
            "train": URI + "mushrooms/mushrooms.train.data",
            "val": URI + "mushrooms/mushrooms.test.data",
        },
        "cwebkb": {
            "train": URI + "cwebkb/cwebkb.train.data",
            "val": URI + "cwebkb/cwebkb.test.data",
        },
        "tmovie": {
            "train": URI + "tmovie/tmovie.train.data",
            "val": URI + "tmovie/tmovie.test.data",
        },
        "adult": {
            "train": URI + "adult/adult.train.data",
            "val": URI + "adult/adult.test.data",
        },
        "cr52": {
            "train": URI + "cr52/cr52.train.data",
            "val": URI + "cr52/cr52.test.data",
        },
        "connect4": {
            "train": URI + "connect4/connect4.train.data",
            "val": URI + "connect4/connect4.test.data",
        },
        "ocr_letters": {
            "train": URI + "ocr_letters/ocr_letters.train.data",
            "val": URI + "ocr_letters/ocr_letters.test.data",
        },
        "rcv1": {
            "train": URI + "rcv1/rcv1.train.data",
            "val": URI + "rcv1/rcv1.test.data",
        },
        "tretail": {
            "train": URI + "tretail/tretail.train.data",
            "val": URI + "tretail/tretail.test.data",
        },
    }

    train_path = os.path.join(data_dir, dataset, f"{dataset}.train.data")
    val_path = os.path.join(data_dir, dataset, f"{dataset}.test.data")
    os.makedirs(os.path.join(data_dir, dataset), exist_ok=True)

    # Download if missing
    if not os.path.exists(train_path):
        urlretrieve(URLS[dataset]["train"], train_path)
    if not os.path.exists(val_path):
        urlretrieve(URLS[dataset]["val"], val_path)

    with urlopen(URLS[dataset]["train"]) as f:
        x_train = np.loadtxt(f, dtype=int, delimiter=",")

    with urlopen(URLS[dataset]["val"]) as f:
        x_val = np.loadtxt(f, dtype=int, delimiter=",")

    x_train = torch.from_numpy(x_train)
    x_val = torch.from_numpy(x_val)

    train_set = torch.utils.data.TensorDataset(x_train, x_train)
    val_set = torch.utils.data.TensorDataset(x_val, x_val)

    return train_set, val_set


def dataset_to_numpy(dataset):
    loader = DataLoader(dataset, batch_size=len(dataset))
    X, _ = next(iter(loader))  # (N, 1, 28, 28)
    return X.view(len(dataset), -1).numpy()  # reshape to (N, 28*28)


if __name__ == "__main__":

    # UCLA
    pbar = tqdm(
        [
            # "nltcs",
            # "msnbc",
            # "kdd",
            # "plants",
            # "jester",
            # "baudio",
            # "bnetflix",
            # "accidents",
            # "retail",
            # "pumsb_star",
            # "dna",
            # "kosarek",
            "msweb",
            "book",
            "tmovie",
            "cwebkb",
            "cr52",
            "c20ng",
            "bbc",
            "ad",
        ]
    )
    for dataset in pbar:
        load_fn = lambda: load_ucla(dataset=dataset)
        if dataset == "mnist":
            load_fn = load_mnist
        train_set, test_set = load_fn()
        X_train = dataset_to_numpy(train_set)
        X_test = dataset_to_numpy(test_set)
        os.makedirs(f"data/{dataset}", exist_ok=True)

        np.save(f"data/{dataset}/train.npy", X_train)
        np.save(f"data/{dataset}/test.npy", X_test)
        pbar.set_postfix({"dataset": dataset})
