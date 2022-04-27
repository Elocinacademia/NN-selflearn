import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
 

#`torch.utils.data.DataLoader` and `torch.utils.data.Dataset`. Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset.

#Download training data from open dataset.
#`torchvision.datasets`module contains `Dataset` objects of real world dataset.
#Every `Dataset` includes two arguments `transform` and `target_transform` to modify the samples and labels.

training_data = datasets.FashionMNIST(
	root = "data",
	train = True,
	download = True,
	transform = ToTensor(),
)

test_data = datasets.FashionMNIST(
	root = "data",
        train = False,,
        download = True,
        transform = ToTensor(),
)

batch_size = 64

#Create data loaders
training_dataloader = Dataloader(training_data, batch_size = batch_size)
test_dataloader = Dataloader(test_data, batch_size = batch_size)

for X,y in test_dataloader:
	print(f"Shape of X [N,C,H,W]: {X.shape}")
	print(f"Shape of y: {y.shape} {y.dtype}")
	break
