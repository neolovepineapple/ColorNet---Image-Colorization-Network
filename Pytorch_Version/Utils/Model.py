from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

class ColorNet(nn.Module):

    def __init__(self):
        super(ColorNet, self).__init__()

        self.encoder_conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=2)
        self.encoder_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.encoder_conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2)
        self.encoder_conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.encoder_conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)
        self.encoder_conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.encoder_conv7 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.encoder_conv8 = nn.Conv2d(512, 256, kernel_size=3, padding=1)

        self.fusion = nn.Conv2d(1256, 256, kernel_size=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.decoder_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.decoder_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.decoder_conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.decoder_conv4 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.decoder_conv5 = nn.Conv2d(16, 2, kernel_size=3, padding=1)



    def forward(self, x, embed):
        output = F.relu(self.encoder_conv1(x))
        output = F.relu(self.encoder_conv2(output))
        output = F.relu(self.encoder_conv3(output))
        output = F.relu(self.encoder_conv4(output))
        output = F.relu(self.encoder_conv5(output))
        output = F.relu(self.encoder_conv6(output))
        output = F.relu(self.encoder_conv7(output))
        output = F.relu(self.encoder_conv8(output))

        embed = embed.unsqueeze(2).unsqueeze(3)
        embed = embed.repeat([1,1,32,32])
        fusion = torch.cat((output, embed), 1) 

        fusion = F.relu(self.fusion(fusion))

        output = self.upsample(F.relu(self.decoder_conv1(fusion)))
        output = self.upsample(F.relu(self.decoder_conv2(output)))
        output = F.relu(self.decoder_conv3(output))
        output = F.relu(self.decoder_conv4(output))
        output = self.upsample(F.relu(self.decoder_conv5(output)))


        return output














