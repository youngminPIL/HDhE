# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from HDhE_test_embedded import Get_test_results_doublehead, Get_test_results_doublehead_inshop
from PIL import Image
import time
import os
from HDhE_model import ft_2Head_512_feature
from tensorboard_logger import configure, log_value
import copy


######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--train_all', action='store_true', help='use all training data')
parser.add_argument('--batchsize', default=40, type=int, help='batchsize')
parser.add_argument('--erasing_p', default=0.5, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--dataset', default='CUB-200', type=str, help='dataset')
parser.add_argument('--stride_f', default='1,1', type=str, help='stride_f')
parser.add_argument('--L4_f', default='2,0', type=str, help='L4_f')
parser.add_argument('--lr_f', default='0.5,4', type=str, help='lr_f')

opt = parser.parse_args()

str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >= 0:
        gpu_ids.append(gid)

inshop = False
sop = False

if opt.dataset == 'CUB-200':
    data_dir = '/home/ro/FG/CUB_RT/pytorch'
elif opt.dataset == 'Cars-196':
    data_dir = '/home/ro/FG/STCAR_RT/pytorch'
elif opt.dataset == 'Inshop':
    inshop = True
    data_dir = '/home/ro/FG/Inshop/pytorch'
elif opt.dataset == 'SOP':
    sop = True
    data_dir = '/home/ro/FG/Stanford_Online_Products/pytorch'


dir_name = '/data/ymro/Access/Double_head/res50_cub/'

lr_factor = []
stride_factor = []
L4id_factor = []

strides = opt.stride_f.split(',')
L4ids = opt.L4_f.split(',')
lrs = opt.lr_f.split(',')
for i in range(2):
    stride_factor.append(int(strides[i]))
    L4id_factor.append(int(L4ids[i]))
    lr_factor.append(float(lrs[i]))

res = 50

e_drop = 20
e_end = 40

f_size = 512
configure(dir_name)
print(dir_name)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

print(gpu_ids[0])
# set gpu ids
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])

######################################################################
# Load Data
# ---------
#


lr1 = 0.1
lr2 = 0.01
lr3 = 0.001
lr4 = 0.0001

init_resize = (256, 256)
resize = (224, 224)

transform_train_list = [
    # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
    transforms.Resize(resize, interpolation=3),
    #transforms.RandomCrop(resize),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

if inshop or sop:
    print('selected directed resizing test')
    transform_val_list = [
        transforms.Resize(resize, interpolation=3),  # Image.BICUBIC
        transforms.CenterCrop(resize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
else:
    transform_val_list = [
        transforms.Resize(init_resize, interpolation=3),  # Image.BICUBIC
        transforms.CenterCrop(resize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]


# print(transform_train_list)
if inshop:
    data_transforms = {
        'train': transforms.Compose(transform_train_list),
        'gallery': transforms.Compose(transform_val_list),
        'query': transforms.Compose(transform_val_list),
    }
else:
    data_transforms = {
        'train': transforms.Compose(transform_train_list),
        'test': transforms.Compose(transform_val_list),
    }


image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
if inshop:
    image_datasets['gallery'] = datasets.ImageFolder(os.path.join(data_dir, 'gallery'), data_transforms['gallery'])
    image_datasets['query'] = datasets.ImageFolder(os.path.join(data_dir, 'query'), data_transforms['query'])
else:
    image_datasets['test'] = datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])


dataloaders = {}

dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=opt.batchsize, shuffle=True, num_workers=16)
if inshop:
    dataloaders['gallery'] = torch.utils.data.DataLoader(image_datasets['gallery'], batch_size= int(opt.batchsize/2), shuffle=False, num_workers=8)
    dataloaders['query'] = torch.utils.data.DataLoader(image_datasets['query'], batch_size= int(opt.batchsize/2), shuffle=False, num_workers=8)
else:
    dataloaders['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size= int(opt.batchsize/2), shuffle=False, num_workers=8)

dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

inputs, classes = next(iter(dataloaders['train']))



def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    for epoch in range(num_epochs):
        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_loss1 = 0.0
            running_corrects1 = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())

                optimizer.zero_grad()

                # forward
                _, _, outputs, outputs1 = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                _, preds1 = torch.max(outputs1.data, 1)

                loss = criterion(outputs, labels)
                loss1 = criterion(outputs1, labels)
                total_loss = loss + loss1
                if phase == 'train':
                    total_loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data
                running_loss1 += loss1.data

                running_corrects += torch.sum(preds == labels.data)
                running_corrects1 += torch.sum(preds1 == labels.data)

            running_corrects = running_corrects.float()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            running_corrects1 = running_corrects1.float()
            epoch_loss1 = running_loss1 / dataset_sizes[phase]
            epoch_acc1 = running_corrects1 / dataset_sizes[phase]

            print('{} Loss: {:.8f} Acc: {:.8f}, Loss1: {:.8f} Acc: {:.8f}'.format(
                phase, epoch_loss, epoch_acc, epoch_loss1, epoch_acc1))

            if phase == 'train':
                log_value('train_loss', epoch_loss, epoch)
                log_value('train_acc', epoch_acc, epoch)

        if inshop:
            results = Get_test_results_doublehead_inshop(image_datasets, dataloaders, model, f_size)
            print(
                'test accuracy : top-1 {:.4f} top-10 {:.4f} top-20 {:.4f} top-30 {:.4f}'.format(results[0] * 100,
                                                                                             results[1] * 100,
                                                                                             results[2] * 100,
                                                                                             results[3] * 100))
        else:
            results = Get_test_results_doublehead(image_datasets['test'], dataloaders['test'], model, f_size, sop)
            if not sop:
                print(
                    'test accuracy : top-1 {:.4f} top-2 {:.4f} top-4 {:.4f} top-8 {:.4f}'.format(results[0] * 100,
                                                                                                 results[1] * 100,
                                                                                                 results[2] * 100,
                                                                                                 results[3] * 100))
            else:
                print(
                    'test accuracy : top-1 {:.4f} top-10 {:.4f} top-100 {:.4f} top-1000 {:.4f}'.format(results[0] * 100,
                                                                                                 results[1] * 100,
                                                                                                 results[2] * 100,
                                                                                                 results[3] * 100))

    return model


####################################################
# Save model
# ---------------------------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join(dir_name, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda(gpu_ids[0])

######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#
def load_network_path(network, save_path):
    network.load_state_dict(torch.load(save_path))
    return network


if not os.path.isdir(dir_name):
    os.mkdir(dir_name)


def set_opt(model):
    params_ft = []
    params_ft.append({'params': model.model.conv1.parameters(), 'lr': lr3*lr_factor[0]})
    params_ft.append({'params': model.model.bn1.parameters(), 'lr': lr3*lr_factor[0]})
    params_ft.append({'params': model.model.layer1.parameters(), 'lr': lr3*lr_factor[0]})
    params_ft.append({'params': model.model.layer2.parameters(), 'lr': lr3*lr_factor[0]})
    params_ft.append({'params': model.model.layer3.parameters(), 'lr': lr3*lr_factor[0]})
    params_ft.append({'params': model.model.layer4.parameters(), 'lr': lr3*lr_factor[0]})
    params_ft.append({'params': model.classifier.parameters(), 'lr': lr2*lr_factor[0]})

    params_ft.append({'params': model.model1.conv1.parameters(), 'lr': lr3*lr_factor[1]})
    params_ft.append({'params': model.model1.bn1.parameters(), 'lr': lr3*lr_factor[1]})
    params_ft.append({'params': model.model1.layer1.parameters(), 'lr': lr3*lr_factor[1]})
    params_ft.append({'params': model.model1.layer2.parameters(), 'lr': lr3*lr_factor[1]})
    params_ft.append({'params': model.model1.layer3.parameters(), 'lr': lr3*lr_factor[1]})
    params_ft.append({'params': model.model1.layer4.parameters(), 'lr': lr3*lr_factor[1]})
    params_ft.append({'params': model.classifier1.parameters(), 'lr': lr2*lr_factor[1]})
    optimizer = optim.SGD(params_ft, momentum=0.9, weight_decay=5e-4, nesterov=True)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=e_drop, gamma=0.1)

    return optimizer, exp_lr_scheduler

criterion = nn.CrossEntropyLoss()

model = ft_2Head_512_feature(int(len(class_names)), f_size, stride=stride_factor, L4id=L4id_factor)
model = model.cuda()

optimizer_ft, exp_lr_scheduler = set_opt(model)

model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=e_end)



