from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

import os
from skimage import io, transform
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
import numpy as np

from Utils import DataGenerator as dGenerator
from Utils import Model, Run



parser = argparse.ArgumentParser(description='Pytorch ColorNet')

parser.add_argument('data', metavar='DIR', help='path to the dataset')

parser.add_argument('val', metavar='DIR', help='path to the validation dataset')


parser.add_argument('action', metavar='Action', type=str, 
        help='train or predict')


parser.add_argument('--epoch', default=500, type=int, metavar= 'N',
        help='number of total epochs to run')


parser.add_argument('--patience', default=20, type=int, metavar= 'N',
        help='stop the epoch earlier when validatation loss doesn\'t drop')


parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')



best_loss = 0

def main():
    global args, best_loss
    args = parser.parse_args()



    model = Model.ColorNet()
    model.cuda()
    criterion = nn.MSELoss().cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.1)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
           print("=> no checkpoint found at '{}'".format(args.resume))


    if args.action = 'train':
        train_loader = dGenerator.makeDataLoader(args.data, 'train')

        val_loader = dGenerator.makeDataLoader(args.val, 'validate')

        for epoch in range(args.epoch):
            print('='*10+'epoch '+str(epoch)+'='*10)

            adjust_learning_rate(optimizer, epoch)

            Run.train(train_loader, model, criterion, optimizer, epoch)

            loss = Run.validate(val_loader, model, criterion)

            print('loss: '+str(loss)+'\n')


            is_best = loss < best_loss

            best_loss = min(loss, best_loss)

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'inception_v3',
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
                }, is_best)
            

        




def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')




if __name__ == '__main__':
    main()







