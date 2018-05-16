from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable

import os
from skimage import io, transform
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
import numpy as np

from Utils import Embedding 

class ColorNetDataset(Dataset):
    def __init__(self, path, transform=None,returnSize=False):
        """
        Args:
            path (string): transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.path = path
        self.imageList = os.listdir(self.path)
        self.transform = transform
        self.returnSize = returnSize

    def __len__(self):
        return len(self.imageList)

    def __getitem__(self, idx):
        imgPath = os.path.join(self.path,self.imageList[idx])
        image = io.imread(imgPath)
        imgSize = image.shape[:2]
        if self.transform:
            image = self.transform(image)
        if self.returnSize:
            return image, imgSize
        else:
            return image 







class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, outputSize):
        assert isinstance(outputSize, (int, tuple))
        self.outputSize = outputSize

    def __call__(self, image):


        h, w = image.shape[:2]
        if isinstance(self.outputSize, int):
            if h > w:
                new_h, new_w = self.outputSize * h / w, self.outputSize
            else:
                new_h, new_w = self.outputSize, self.outputSize * w / h
        else:
            new_h, new_w = self.outputSize

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))


        return img




class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        
        return image


def ToTensor(batch):
    batch = batch.transpose((0, 3, 1, 2))
    return torch.from_numpy(batch).float()

def makeDataLoader(path, category, batch_size=20):
    """
        Data Generator With Default Setting 
    """

    def batchSplit(batch, isfloat):
            numpy_batch=batch.data.numpy()
            if not isfloat:
                numpy_batch /= 255.0

            grayscaled_rgb = gray2rgb(rgb2gray(numpy_batch))
            embed = Embedding.creatEmbedding(grayscaled_rgb)
            lab_batch = rgb2lab(numpy_batch)
            X_batch = lab_batch[:,:,:,0]
            X_batch = ToTensor(X_batch.reshape(X_batch.shape+(1,)))
            Y_batch = ToTensor(lab_batch[:,:,:,1:] / 128)
            return [X_batch.float(), embed.float()], Y_batch.float()


    if category == 'train':
        dataSet = ColorNetDataset(path=path,
            transform=transforms.Compose([Rescale((300,300)),
                                               RandomCrop((256,256))]))



        dataLoader = DataLoader(dataSet, batch_size=20,
                        shuffle=True, num_workers=4)
        isfloat = (dataSet[0].max() <= 1)
        for batch in dataLoader:
            yield batchSplit(batch, isfloat)

    if category == 'validate':
        dataSet = ColorNetDataset(path=path,
            transform=transforms.Compose([Rescale((256,256))]))



        dataLoader = DataLoader(dataSet, batch_size=20,
                        shuffle=False, num_workers=4)
        isfloat = (dataSet[0].max() <= 1)
        for batch in dataLoader:
            yield batchSplit(batch, isfloat)

    if category == 'test':
        dataSet = ColorNetDataset(path=path,
            transform=transforms.Compose([Rescale((256,256))]),
            returnSize=True)
        batchSize = min(20, len(dataSet)) 
        dataLoader = DataLoader(dataSet, batch_size=batchSize,
                        shuffle=False, num_workers=4)
        isfloat = (dataSet[0][0].max() <= 1)
        for batch, size in dataLoader:
            [X_batch, embed], Y_batch = batchSplit(batch, isfloat)
            yield [X_batch, embed], size

    if category == 'original':
        dataSet = ColorNetDataset(path=path,
            transform=transforms.Compose([Rescale((256,256))]))



        dataLoader = DataLoader(dataSet, batch_size=20,
                        shuffle=False, num_workers=4)
        isfloat = (dataSet[0].max() <= 1)
        for batch in dataLoader:
            yield (batch.float())







            



        



        





















