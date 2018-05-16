from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import torchvision.models as models

import numpy as np
from skimage.transform import resize
model = models.__dict__['inception_v3'](pretrained=True)

def ToTensor(batch):
    batch = batch.transpose((0, 3, 1, 2))
    return torch.from_numpy(batch).float()



def creatEmbedding(batchs):
    batch_resize = []
    for i in batchs:
        i = resize(i, (299, 299, 3), mode='constant')
        batch_resize.append(i)
    batch_resize = ToTensor(np.array(batch_resize))

    output = model(batch_resize)[0]
    return output






