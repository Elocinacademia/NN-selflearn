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
