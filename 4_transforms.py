

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root = 'data',
    train = True,
    download = False,
    transform = ToTensor(),
    target_transform = Lambda(lambda y: torch.zeros(10, dtype = toech.float).scatter_(0,torch.tensor(y),value =1))
    )


'''
PyTorch 的图像处理库torchvision，里面有一个函数 names transforms. 这个函数集成了随机翻转、旋转、增强对比度、转化为tensor、转化为图像等功能。

这里我们主要关注toTensor() 因为目前还不需要处理图像相关的data
transform = transforms.Compose([
    transforms.ToTensor(), #将图片变成tensor，并且把数值normalize到[0,1]
    ])

使用时可直接img = transform(x)

transforms还可以图片翻转，处理噪声什么的
'''


# ToTensor() converts a PIL image or NumPy ndarray into a FloatTensor. and scales the images's pixel intensity values in the range [0.,1.]

# Lambda Transforms

'''
Lambda transforms apply any user-defined lambda function. Here, we define a function to turn the integer into a one-hot encoded tensor. It first creates a zero tensor of size 10 (the number of labels in our dataset) and calls scatter_ which assigns a value=1 on the index as given by the label y.
'''

# One-hot encoding: 一位有效编码，采用位状态寄存器来对个状态进行编码
# In ML 特征有时候并不总是连续值，有可能是分类值 e.g., male and female 在ML中通常要对这样的特征进行特征数字化
# Note：我们可以直接把性别数字化为[0,1,3]，但这样的特征处理并不能直接放入机器学习算法中

# 具体做法：假设词典中不同字符的数量为N，每个字符已经同一个从0到N-1的连续整数值索引一一对应。如果一个字符的索引是整数i，那么就创建一个全0的长为N的向量，并将其位置为i的元素设成1.该向量就是对原字符的one-hot向量！！！

target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))




import pdb;pdb.set_trace()

