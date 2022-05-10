import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt



# Only need to download the dataset once at the beginning (set 'download' = True)
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=False,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=False,
    transform=ToTensor()
)

# Iterating and visualizing the dataset

labels_map ={
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

figure = plt.figure(figsize = (8,8))
cols,rows = 3,3
for i in range (1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size = (1,)).item()
    img, label = training_data[sample_idx]  
    figure.add_subplot(rows,cols,i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap = 'gray')
#plt.show()

# Creating a custom dataset for own files:
# A custom dataset class must implement three functions: __init__, __len__, and __getitem__
# The FashionMNIST images are stored in a directory img_dir, and their labels are stored separately in a CSV file annotations_file.

import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self,annotations_file,img_dir,transform=None, target_transform=None):
        # the __init__ function is run once when instantiating the dataset object, we initialize the directory containing images, the annotations file, and both transforms
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        # this function returns the number of samples in our dataset
        return len(self.img_labels)
    def __getitem__(self,idx):
        # this function loads and returns a sample from the dataset at the given index `idx`. Based on the index, it identifies the images' location on disk, converts that to a tensor using `read_image`, retrieves the corresponding label from the csv data in `self.img_labels`, calls the transform functions on them(if applicable), and returns the tensor image and corresponding label in a tuple.
        img_path = os.path.join(self.img_dir,self.img_labels.iloc[idx,0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx,1]
        if self.transform:
            img = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label







