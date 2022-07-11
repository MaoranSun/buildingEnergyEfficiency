import argparse
import os
import random
import shutil
import time
import datetime
import warnings
import sys

import numpy as np
import pandas as pd
from PIL import Image
import nonechucks as nc

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data as data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision import transforms as trn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score

# global variable
date_today = str(datetime.datetime.now().date()) 
best_acc1 = 0

# parse argument
parser = argparse.ArgumentParser(description='PyTorch multiBranch Training')

parser.add_argument('data', metavar='DIR',
                    help='dataframe path to dataset ' 'include: index path label split')
parser.add_argument('-c', '--num-classes', required = True, type=int, metavar='N',
                    help='number of categories for classification')
parser.add_argument('-inDIM', type=int, metavar='N', default=5,
                    help='dimension of second branch input')
parser.add_argument('-outDIM', type=int, metavar='N', default=48,
                    help='dimension of second branch output')
parser.add_argument('-s', '--save-folder', default = './model', metavar='PATH', type=str,
                    help='folder path to save model files and log files (default: ./model)')
parser.add_argument('-m', '--save-prefix', default = date_today, metavar='PATH', type=str,
                    help='prefix string to specify version or task type  (default: [date])')
parser.add_argument('-l', '--resume-log', default= '', metavar='PATH', type=str, 
                    help='path of log file to append (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate',  action='store_true',
                    help='evaluate model on validation set')
parser.add_argument(      '--evaluate-feature',  action='store_true',
                    help='whether save deep feature in evaluate model')
parser.add_argument(      '--evaluate-split', default=1, type=int,
                    help='how many splits for DataFrame in evaluation mode')
parser.add_argument('-p', '--pretrained', dest='pretrained',  action='store_true',
                    help='use pre-trained model')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--safe-load',  action='store_true',
                    help='way to build data loader---handle decayed samples or not')
parser.add_argument('--img-size', default=[224,224], type=list,
                    help='size to resize image')
parser.add_argument('--lr-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate (default: 0.0001)', dest='lr')
parser.add_argument('--lr-deGamma', default=0.1, type=float,
                    metavar='LR', help='lr-deGamma (default: 0.1)')
parser.add_argument('--lr-deStep', default=50, type=int,
                    metavar='LR', help='lr-deStep (default: 50)')  
    
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--topk', default=3, type=int,
                    help='print topk accuracy (default: 3)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default= '1,2,3,4', type=str,
                    help='GPU id(s) to use.'
                         'Format: a comma-delimited list')
parser.add_argument('--imbalanced', action="store_true",
                    help='use weighted random sampler for dataloader')

def main():
    args = parser.parse_args()
    print(args) 
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if len(args.gpu) is not 0:
        print( 'You have chosen a specific GPU(s) : {}'.format(args.gpu))
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
        os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
        
    if not os.path.exists(args.save_folder) :
        os.makedirs(args.save_folder)
        
    main_worker(args)

def main_worker(args):
    global best_acc1

    # create model
    print('creating multibranch model')
    model = multiBranchModel(args.inDIM, args.outDIM, args.num_classes)
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), 
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    tf = trn.Compose([
            trn.Resize(args.img_size),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            print("Current accuracy in validation: {}".format(checkpoint['best_acc1']))
            best_acc1=0
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return                
            
    if args.evaluate:
        if os.path.isfile(args.resume):
            evaluate(model, criterion, tf, args)
            return
        else:
            print( 'Evaluation mode without a pretrained model, quit.'   ) 
            return

    # Data loading code
    dataDF = pd.read_pickle(args.data)
    trainDF = dataDF[dataDF.split == 'train']
    valDF = dataDF[dataDF.split == 'val']

    # deal with imbalanced dataset with weighted random sampler
    if args.imbalanced:
        class_counts = trainDF.label.value_counts().sort_index().values
        num_samples = sum(class_counts)
        labels = trainDF.label.values

        class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
        weights = [class_weights[labels[i]] for i in range(int(num_samples))]
        sampler = data.WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))
    
    args.batch_num_val = len(valDF)/args.batch_size
    args.batch_num_train = len(trainDF)/args.batch_size
    
    # if args.imbalanced:
    #     trainDataset = Dataset(trainDF.path.tolist(), trainDF['attrs'].tolist(), np.array(trainDF.label), tf, sampler = sampler)
    # else:
    trainDataset = Dataset(trainDF.path.tolist(), trainDF['attrs'].tolist(), np.array(trainDF.label), tf)

    valDataset = Dataset(valDF.path.tolist(), valDF['attrs'].tolist(), np.array(valDF.label), tf)

    if args.safe_load is True:
        trainDataset = nc.SafeDataset(trainDataset)
        train_loader = nc.SafeDataLoader(
            trainDataset,
            batch_size= args.batch_size, 
            shuffle=False,
            num_workers= args.workers, 
            pin_memory=True)
        
        valDataset = nc.SafeDataset(valDataset)
        val_loader = nc.SafeDataLoader(
            valDataset,
            batch_size= args.batch_size, 
            shuffle=False,
            num_workers= args.workers, 
            pin_memory=True)
    else:
        if args.imbalanced:
            train_loader = torch.utils.data.DataLoader(
                    trainDataset,
                    batch_size= args.batch_size, 
                    shuffle=False,
                    num_workers= args.workers, 
                    pin_memory=True,
                    sampler=sampler)
        else:
            train_loader = torch.utils.data.DataLoader(
                    trainDataset,
                    batch_size= args.batch_size, 
                    shuffle=True,
                    num_workers= args.workers, 
                    pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
                valDataset,
                batch_size= args.batch_size, 
                shuffle=False,
                num_workers= args.workers, 
                pin_memory=True)

    logList = []
    if len(args.resume_log):
        logList.append(pd.read_pickle(args.resume_log))
        
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
       
        # train for one epoch
        logEpochDF = train(train_loader, model, criterion, optimizer, epoch, args)
        logList.append(logEpochDF)

        # evaluate on validation set
        logEpochDF, acc1 = validate(val_loader, model, criterion, epoch, args)
        logList.append(logEpochDF)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_model_name =  args.save_prefix +'_' + str(epoch) + '_checkpoint.pth.tar'

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.save_folder, save_model_name )
        
        save_log_name = args.save_prefix + '_log.p'
        pd.concat(logList).reset_index(drop=True).to_pickle(os.path.join(args.save_folder, save_log_name))

def train(train_loader, model, criterion, optimizer, epoch, args):

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    
    logList = []
    for i, (image, attr, target) in enumerate(train_loader):

        image = image.cuda()
        attr = attr.cuda()
        target = target.cuda()
#         print('target train 0/1: {}/{}'.format(
#               len(torch.where(target == 0)[0]), len(torch.where(target == 3)[0])))

        # compute output
        output = model(image, attr)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, args.topk))
        losses.update(loss.item(), image.size(0))
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        date_time = str(datetime.datetime.now()) 
        
        stepLogDict = {'time':date_time,
                       'type':'train',
                       'epoch':epoch,
                       'step':i, 
                       'loss':float(losses.val),
                       'top1':float(top1.val),
                       'top5':float(top5.val)}
        logList.append(stepLogDict)
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {date_time}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, args.batch_num_train, date_time= date_time,
                      loss=losses, top1=top1, top5=top5))

    logEpochDF = pd.DataFrame(logList)
    return logEpochDF

def validate(val_loader, model, criterion, epoch, args):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (image, attr, target) in enumerate(val_loader):
            print(str(datetime.datetime.now()))
            image = image.cuda()
            attr = attr.cuda()
            target = target.cuda()

            # compute output
            output = model(image, attr)
            print(str(datetime.datetime.now()))
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, args.topk))
            losses.update(loss.item(), image.size(0))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            
            date_time = str(datetime.datetime.now())
            
            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {date_time}\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, args.batch_num_val, date_time= date_time,
                          loss=losses, top1=top1, top5=top5))    
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        
    stepLogDict = {'time':date_time,
                   'type':'val',
                   'epoch':epoch,
                   'step':None, 
                   'loss':float(losses.avg),
                   'top1':float(top1.avg),
                   'top5':float(top5.avg)}
    
    logEpochDF = pd.DataFrame(stepLogDict, index = [0])
    return logEpochDF, top1.avg

def evaluate(model, criterion, transformer, args):
    
    evaluateDF = pd.read_pickle(args.data).reset_index(drop=True)
    evaluateDF['predict'] = None 
    evaluateDF['prob'] = None 

    imglist = evaluateDF.path.tolist()
    attrsList = evaluateDF['attrs'].tolist()
    indexlist = evaluateDF.index.tolist()

    dataset = EvaluateDataset(imglist, attrsList, indexlist, transformer)
    loader = data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            shuffle=False)
 
    # switch to evaluate mode
    model.eval()
     
    num_batches = len(evaluateDF) / args.batch_size
    with torch.no_grad():
        for batch_idx, (image, attr, paths, img_indexes) in enumerate(loader):

            print(str(datetime.datetime.now()), ' %d / %d' % (batch_idx, num_batches)) 

            image = image.cuda()
            attr = attr.cuda()
            #input = torch.autograd.Variable(input, volatile = True).cuda()

            logits = model(image, attr)

            h_x = torch.nn.functional.softmax(logits, 1).data.squeeze()
            h_x = h_x.reshape(-1, args.num_classes)
            allProbs = h_x.cpu().numpy().round(3)
            probs, idx = h_x.sort(1, True)
            probs = probs[:,0].cpu().numpy()
            idx = idx[:,0].cpu().numpy()

            batchDF = pd.DataFrame(dict(prob=probs, predict = idx))
            batchDF['path'] = paths
            batchDF.index = img_indexes.numpy()

            evaluateDF.update(batchDF)
    
    trainDF = evaluateDF[evaluateDF.split =='train']
    valDF = evaluateDF[evaluateDF.split =='val']

    totalAcc = accuracy_score(evaluateDF.label.tolist(), evaluateDF.predict.tolist())
    trainAcc = accuracy_score(trainDF.label.tolist(), trainDF.predict.tolist())
    valAcc = accuracy_score(valDF.label.tolist(), valDF.predict.tolist())
    
    save_path = os.path.join(args.save_folder, args.save_prefix +'_evaluate.p')
    evaluateDF.to_pickle(save_path)
        
    print( 'Evaluation file was saved at ', save_path)
    print( 'Acc train: ', trainAcc)
    print( 'Acc val: ', valAcc)
    print( 'Acc total: ', totalAcc)
    
#===================== Utils =====================#
class multiBranchModel(nn.Module):
    def __init__(self, in_dim, out_dim, num_classes):
        super(multiBranchModel, self).__init__()
        # densenet branch functions
        self.cnn = models.__dict__['densenet121'](pretrained=True)._modules.get('features')
        # ann branch functions
        self.neural1 = nn.Linear(in_dim, 1024)
        self.neural2 = nn.Linear(1024, 512)
        self.neural3 = nn.Linear(512, 256)
        self.neural4 = nn.Linear(256, out_dim)
        
        self.fc1 = nn.Linear(1024 + out_dim, 1024)
        self.fc2 = nn.Linear(1024, 256)
        # self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, num_classes)
        
    def forward(self, image, data):
        # densenet branch
        x1 = self.cnn(image)
        x1 = F.relu(x1, inplace=True)
        x1 = F.adaptive_avg_pool2d(x1, (1,1))
        x1 = torch.flatten(x1, 1)
        
        # ann branch
        x2 = F.relu(self.neural1(data))
        x2 = F.relu(self.neural2(x2))
        x2 = F.relu(self.neural3(x2))
        x2 = F.relu(self.neural4(x2))
        
        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class Dataset(data.Dataset):
    def __init__(self, imgList, attrList, labelList, transform = None):
     
        self.imgList = imgList
        self.attrList = attrList
        self.labelList = labelList
        self.transform = transform
            
    def __getitem__(self, index):

        img_path = self.imgList[index]
        image = Image.open(img_path).convert('RGB')
        attrs = self.attrList[index]
        label = self.labelList[index]
    
        if self.transform:
            image = self.transform(image)
        
        attrs = torch.FloatTensor(attrs)
        sample = (image,attrs,label)
        
        return sample
        
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.imgList) 

class EvaluateDataset(data.Dataset):
    def __init__(self, imgList, attrList, imgindexList, transform = None):
     
        self.imgList = imgList
        self.attrList = attrList
        self.transform = transform
        self.imgindexList = imgindexList
        
    def __getitem__(self, index):

        img_path = self.imgList[index]
        attrs = self.attrList[index]
        attrs = torch.FloatTensor(attrs)
        img_index = self.imgindexList[index]
        
        image = Image.open(img_path).convert('RGB')
    
        if self.transform:
            image = self.transform(image)
        
        return image, attrs, img_path, img_index
        
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.imgList) 

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (args.lr_deGamma ** (epoch // args.lr_deStep))
 
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
        
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_checkpoint(state, is_best, folder = './', filename='checkpoint.pth.tar'):
    save_path = os.path.join(folder, filename)
    torch.save(state,  save_path)
    if is_best:
        shutil.copyfile(save_path, os.path.join(folder, 'model_best.pth.tar')  )

    
if __name__ == '__main__':
    main()