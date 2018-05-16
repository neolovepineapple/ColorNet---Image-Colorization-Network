import time


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models



def train(train_loader, model, criterion, optimizer, epoch, print_freq=20):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()

    for i, [X, embed], Y in enumerate(train_loader):

        data_time.update(time.time() - end)

        X = X.cuda()
        embed = embed.cuda()
        Y = Y.cuda()

        output = model(X, embed)

        loss = criterion(output, Y)

        losses.update(loss.item(), X.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))


def validate(val_loader, model, criterion):

    batch_time = AverageMeter()
    losses = AverageMeter()
    model.eval()

    with torch.no_grad():
        end = time.time()

        for i, [X, embed], Y in enumerate(val_loader):

            output = model(X, embed)

            loss = criterion(output, Y)

            losses.update(loss.item(), X.size(0))

            batch_time.update(time.time() - end)
            end = time.time()


            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses))

    return losses.avg



 






