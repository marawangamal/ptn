import torch
import torchvision
import torchvision.transforms as transforms

import os, certifi
from mtp.mheads._abc import AbstractDisributionHeadConfig
from mtp.mheads import MHEADS

os.environ["SSL_CERT_FILE"] = certifi.where()

# Make the image binary: threshold at 0.5 after ToTensor, then convert to long (int)
transform = transforms.Compose([transforms.ToTensor(), lambda x: (x > 0.5).long()])

# Create datasets for training & validation, download if necessary
training_set = torchvision.datasets.MNIST(
    "./data", train=True, transform=transform, download=True
)
validation_set = torchvision.datasets.MNIST(
    "./data", train=False, transform=transform, download=True
)

# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True)
validation_loader = torch.utils.data.DataLoader(
    validation_set, batch_size=4, shuffle=False
)

# Report split sizes
print("Training set has {} instances".format(len(training_set)))
print("Validation set has {} instances".format(len(validation_set)))

model = MHEADS["moe_proj"](
    AbstractDisributionHeadConfig(
        horizon=784,
        d_model=10,  # 9 digits
        d_output=2,  # 2 classes
        rank=32,
    )
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for batch in training_loader:
    y, x = batch  # for gen modelling reverse x, y
    B = x.shape[0]
    z = (
        torch.nn.functional.one_hot(
            x,
            num_classes=10,
        )
        .reshape(B, -1)
        .to(torch.float32)
    )  # (B, 10)
    z, y = z.to(device), y.to(device)
    print(model(z, y.reshape(B, -1)))
