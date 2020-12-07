import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
# import modelym
from torch.autograd import Variable
import torch.nn.functional as F
import copy
import numpy as np

# torch.manual_seed(701)
# torch.cuda.manual_seed(701)

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=True, relu=True, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(  p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x


class ClassBlock_feature(nn.Module):
    def __init__(self, input_dim, class_num, dropout=True, relu=True, num_bottleneck=512):
        super(ClassBlock_feature, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        add_block2 = []
        if relu:
            add_block2 += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block2 += [nn.Dropout(p=0.5)]
        add_block2 = nn.Sequential(*add_block2)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.add_block2 = add_block2
        self.classifier = classifier

    def forward(self, x):
        f = self.add_block(x)
        x = self.add_block2(f)
        x = self.classifier(x)
        return f, x

class ClassBlock_BNLinear(nn.Module):
    def __init__(self, input_dim, class_num):
        super(ClassBlock_BNLinear, self).__init__()
        add_block = []
        add_block += [nn.BatchNorm1d(input_dim)]
        add_block += [nn.Linear(input_dim, class_num)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_classifier)
        self.add_block = add_block

    def forward(self, x):
        x = self.add_block(x)
        return x

class ClassBlock_BNLinear_noModule(nn.Module):
    def __init__(self, input_dim, class_num):
        super(ClassBlock_BNLinear_noModule, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.fc = nn.Linear(input_dim, class_num)
        self.fc.apply(weights_init_classifier)

    def forward(self, x):
        x = self.bn(x)
        x = self.fc(x)
        return x

class ClassBlock_2dim_BNLinear_noModule(nn.Module):
    def __init__(self, input_dim, class_num):
        super(ClassBlock_2dim_BNLinear_noModule, self).__init__()

        self.bn1 = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, 2)

        self.bn2 = nn.BatchNorm1d(2)
        self.fc2 = nn.Linear(2, class_num)
        self.fc1.apply(weights_init_kaiming)
        self.fc2.apply(weights_init_classifier)

    def forward(self, x):
        x = self.bn1(x)
        f = self.fc1(x)

        x = self.bn2(f)
        x = self.fc2(x)
        return f, x




# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock_direct(nn.Module):
    def __init__(self, input_dim, class_num, dropout=True, relu=True, num_bottleneck=512):
        super(ClassBlock_direct, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x





class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, conv1, bn1, conv2, bn2, downsample, stride):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1
        self.bn1 = bn1
        self.conv2 = conv2
        self.bn2 = bn2
        # self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    # def forward(self, x):
    #     # residual = x
    #
    #     out = self.conv1(x)
    #     out = self.bn1(out)
    #     out = self.relu(out)
    #
    #     out = self.conv2(out)
    #     out = self.bn2(out)
    #     out = self.relu(out)
    #
    #     # out = self.conv3(out)
    #     # out = self.bn3(out)
    #
    #     if self.downsample is not None:
    #         residual = self.downsample(x)
    #
    #     # out += residual
    #     out = self.relu(out)
    #
    #     return out

    def forward(self, x):
        # residual = x
        residual_exist = False
        out = self.conv1(x)
        if self.downsample is not None:
            residual = self.bn1(out)
            out = self.relu(residual)
            residual_exist = True

        else:
            out = self.bn1(out)
            out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # out = self.conv3(out)
        # out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            residual_exist = True

        if residual_exist:
            out += residual

        out = self.relu(out)

        return out





class ft_net_option_feature(nn.Module):

    def __init__(self, class_num, f_size=512, stride=2, L4id=2):
        super(ft_net_option_feature, self).__init__()
        self.L4id = L4id
        model_ft = models.resnet50(pretrained=True)
        feature_size = f_size

        # avg pooling to global+ pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # model_ft.layer4[2] = Bottleneck(model_ft.layer4[2].conv1, model_ft.layer4[2].bn1, model_ft.layer4[2].conv2,
        #                              model_ft.layer4[2].bn2, model_ft.layer4[2].downsample, model_ft.layer4[2].stride)

        if L4id == 0:
            # model_ft.layer4[L4id] = Bottleneck(model_ft.layer4[L4id].conv1, model_ft.layer4[L4id].bn1,
            #                                    model_ft.layer4[L4id].conv2,
            #                                    model_ft.layer4[L4id].bn2, None,
            #                                    model_ft.layer4[L4id].stride)
            if stride == 1:
                model_ft.layer4[0].conv2.stride = (1,1)
        else:
            # model_ft.layer4[L4id] = Bottleneck(model_ft.layer4[L4id].conv1, model_ft.layer4[L4id].bn1,
            #                                    model_ft.layer4[L4id].conv2,
            #                                    model_ft.layer4[L4id].bn2, model_ft.layer4[L4id].downsample,
            #                                    model_ft.layer4[L4id].stride)
            if stride == 1:
                model_ft.layer4[0].downsample[0].stride = (1,)
                model_ft.layer4[0].conv2.stride = (1,1)

        # self.pooling = CompactBilinearPooling(feature_size*4, feature_size*4, feature_size)
        self.model = model_ft
        # self.classifier = ClassBlock_feature(feature_size, class_num, num_bottleneck = f_size)
        # self.classifier1 = ClassBlock_feature(feature_size, class_num, num_bottleneck = f_size)
        # self.classifier = ClassBlock_feature(feature_size, class_num)
        #self.classifier = ClassBlock_BNLinear(feature_size, class_num)
        self.classifier = ClassBlock(feature_size, class_num)


    #
    # def get_gradient_vector(self):
    #     gradient = self.model.layer4[0].conv2.weight.grad.cpu()


    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        if self.L4id == 0:
            x = self.model.layer4[0](x)
        elif self.L4id == 1:
            x = self.model.layer4[0](x)
            x = self.model.layer4[1](x)
        else:
            x = self.model.layer4(x)
        x = self.model.avgpool(x)
        f0 = torch.squeeze(x)
        x = self.classifier(f0)

        # x = torch.squeeze(x)
        # f0, x = self.classifier(x)
        return f0, x


class ft_2Head_512_feature(nn.Module):

    def __init__(self, class_num, f_size, stride, L4id):
        # class_num: # of target dataset's class
        # f_size = 512
        # 4_1 layer의 stride 옵션, 기본적으로 2이지만 1로 바꿀수 있음
        # 레이어 4 중에서 어디까지 쓸것인지 [0,1,2] 0이 4_1까지만 쓰는 옵션

        super(ft_2Head_512_feature, self).__init__()
        self.L4id = L4id
        feature_size = f_size

        model_ft = models.resnet50(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if L4id[0] == 0:
            model_ft.layer4[L4id[0]] = Bottleneck(model_ft.layer4[L4id[0]].conv1, model_ft.layer4[L4id[0]].bn1,
                                               model_ft.layer4[L4id[0]].conv2,
                                               model_ft.layer4[L4id[0]].bn2, None,
                                               model_ft.layer4[L4id[0]].stride)
            if stride[0] == 1:
                model_ft.layer4[0].conv2.stride = (1,1)
        else:
            model_ft.layer4[L4id[0]] = Bottleneck(model_ft.layer4[L4id[0]].conv1, model_ft.layer4[L4id[0]].bn1,
                                               model_ft.layer4[L4id[0]].conv2,
                                               model_ft.layer4[L4id[0]].bn2, model_ft.layer4[L4id[0]].downsample,
                                               model_ft.layer4[L4id[0]].stride)
            if stride[0] == 1:
                model_ft.layer4[0].downsample[0].stride = (1,)
                model_ft.layer4[0].conv2.stride = (1,1)
        self.model = model_ft
        self.classifier = ClassBlock(feature_size, class_num)

        model_ft1 = models.resnet50(pretrained=True)
        model_ft1.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if L4id[1] == 0:
            model_ft1.layer4[L4id[1]] = Bottleneck(model_ft1.layer4[L4id[1]].conv1, model_ft1.layer4[L4id[1]].bn1,
                                                   model_ft1.layer4[L4id[1]].conv2,
                                                   model_ft1.layer4[L4id[1]].bn2, None,
                                                   model_ft1.layer4[L4id[1]].stride)
            if stride[1] == 1:
                model_ft1.layer4[0].conv2.stride = (1,1)
        else:
            model_ft1.layer4[L4id[1]] = Bottleneck(model_ft1.layer4[L4id[1]].conv1, model_ft1.layer4[L4id[1]].bn1,
                                                   model_ft1.layer4[L4id[1]].conv2,
                                                   model_ft1.layer4[L4id[1]].bn2, model_ft1.layer4[L4id[1]].downsample,
                                                   model_ft1.layer4[L4id[1]].stride)
            if stride[1] == 1:
                model_ft1.layer4[0].downsample[0].stride = (1,)
                model_ft1.layer4[0].conv2.stride = (1,1)
        self.model1 = model_ft1
        self.classifier1 = ClassBlock(feature_size, class_num)


    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        # # layer 2 까지 쉐어하고 3부터 분기하는 구조임 필요에 따라서 이부분을 바꾸어 분기하는 layer를 조절할수 있음.

        # x0 = self.model.conv1(x)
        # x0 = self.model.bn1(x0)
        # x0 = self.model.relu(x0)
        # x0 = self.model.maxpool(x0)
        # x0 = self.model.layer1(x0)
        x = self.model.layer2(x)

        x0 = self.model.layer3(x)
        if self.L4id[0] == 0:
            x0 = self.model.layer4[0](x0)
        elif self.L4id[0] == 1:
            x0 = self.model.layer4[0](x0)
            x0 = self.model.layer4[1](x0)
        else:
            x0 = self.model.layer4(x0)
        x0 = self.model.avgpool(x0)
        f0 = torch.squeeze(x0)
        x0 = self.classifier(f0)


        # x1 = self.model1.conv1(x)
        # x1 = self.model1.bn1(x1)
        # x1 = self.model1.relu(x1)
        # x1 = self.model1.maxpool(x1)
        # x1 = self.model1.layer1(x1)

        # x1 = self.model1.layer2(x1)
        x1 = self.model1.layer3(x)
        if self.L4id[1] == 0:
            x1 = self.model1.layer4[0](x1)
        elif self.L4id[1] == 1:
            x1 = self.model1.layer4[0](x1)
            x1 = self.model1.layer4[1](x1)
        else:
            x1 = self.model1.layer4(x1)
        x1 = self.model1.avgpool(x1)
        f1 = torch.squeeze(x1)
        x1 = self.classifier1(f1)

        # x = self.model.layer3(x)
        #
        # if self.L4id[0] == 0:
        #     x0 = self.model.layer4[0](x)
        # elif self.L4id[0] == 1:
        #     x0 = self.model.layer4[0](x)
        #     x0 = self.model.layer4[1](x0)
        # else:
        #     x0 = self.model.layer4(x)
        # x0 = self.model.avgpool(x0)
        # f0 = torch.squeeze(x0)
        # x0 = self.classifier(f0)
        #
        # if self.L4id[1] == 0:
        #     x1 = self.model1.layer4[0](x)
        # elif self.L4id[1] == 1:
        #     x1 = self.model1.layer4[0](x)
        #     x1 = self.model1.layer4[1](x1)
        # else:
        #     x1 = self.model1.layer4(x)
        # x1 = self.model1.avgpool(x1)
        # f1 = torch.squeeze(x1)
        # x1 = self.classifier1(f1)

        return f0, f1, x0, x1


class ft_3Head_512_feature(nn.Module):

    def __init__(self, class_num, f_size, stride, L4id):
        super(ft_3Head_512_feature, self).__init__()
        self.L4id = L4id
        feature_size = f_size

        model_ft = models.resnet50(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if L4id[0] == 0:
            model_ft.layer4[L4id[0]] = Bottleneck(model_ft.layer4[L4id[0]].conv1, model_ft.layer4[L4id[0]].bn1,
                                               model_ft.layer4[L4id[0]].conv2,
                                               model_ft.layer4[L4id[0]].bn2, None,
                                               model_ft.layer4[L4id[0]].stride)
            if stride[0] == 1:
                model_ft.layer4[0].conv2.stride = (1,1)
        else:
            model_ft.layer4[L4id[0]] = Bottleneck(model_ft.layer4[L4id[0]].conv1, model_ft.layer4[L4id[0]].bn1,
                                               model_ft.layer4[L4id[0]].conv2,
                                               model_ft.layer4[L4id[0]].bn2, model_ft.layer4[L4id[0]].downsample,
                                               model_ft.layer4[L4id[0]].stride)
            if stride[0] == 1:
                model_ft.layer4[0].downsample[0].stride = (1,)
                model_ft.layer4[0].conv2.stride = (1,1)
        self.model = model_ft
        self.classifier = ClassBlock(feature_size, class_num)

        model_ft1 = models.resnet50(pretrained=True)
        model_ft1.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if L4id[1] == 0:
            model_ft1.layer4[L4id[1]] = Bottleneck(model_ft1.layer4[L4id[1]].conv1, model_ft1.layer4[L4id[1]].bn1,
                                                   model_ft1.layer4[L4id[1]].conv2,
                                                   model_ft1.layer4[L4id[1]].bn2, None,
                                                   model_ft1.layer4[L4id[1]].stride)
            if stride[1] == 1:
                model_ft1.layer4[0].conv2.stride = (1,1)
        else:
            model_ft1.layer4[L4id[1]] = Bottleneck(model_ft1.layer4[L4id[1]].conv1, model_ft1.layer4[L4id[1]].bn1,
                                                   model_ft1.layer4[L4id[1]].conv2,
                                                   model_ft1.layer4[L4id[1]].bn2, model_ft1.layer4[L4id[1]].downsample,
                                                   model_ft1.layer4[L4id[1]].stride)
            if stride[1] == 1:
                model_ft1.layer4[0].downsample[0].stride = (1,)
                model_ft1.layer4[0].conv2.stride = (1,1)
        self.model1 = model_ft1
        self.classifier1 = ClassBlock(feature_size, class_num)


        model_ft2 = models.resnet50(pretrained=True)
        model_ft2.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if L4id[2] == 0:
            model_ft2.layer4[L4id[2]] = Bottleneck(model_ft2.layer4[L4id[2]].conv1, model_ft2.layer4[L4id[2]].bn1,
                                                   model_ft2.layer4[L4id[2]].conv2,
                                                   model_ft2.layer4[L4id[2]].bn2, None,
                                                   model_ft2.layer4[L4id[2]].stride)
            if stride[2] == 1:
                model_ft2.layer4[0].conv2.stride = (1,1)
        else:
            model_ft2.layer4[L4id[2]] = Bottleneck(model_ft2.layer4[L4id[2]].conv1, model_ft2.layer4[L4id[2]].bn1,
                                                   model_ft2.layer4[L4id[2]].conv2,
                                                   model_ft2.layer4[L4id[2]].bn2, model_ft2.layer4[L4id[2]].downsample,
                                                   model_ft2.layer4[L4id[2]].stride)
            if stride[2] == 1:
                model_ft2.layer4[0].downsample[0].stride = (1,)
                model_ft2.layer4[0].conv2.stride = (1,1)
        self.model2 = model_ft2
        self.classifier2 = ClassBlock(feature_size, class_num)


    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)

        x0 = self.model.layer3(x)
        if self.L4id[0] == 0:
            x0 = self.model.layer4[0](x0)
        elif self.L4id[0] == 1:
            x0 = self.model.layer4[0](x0)
            x0 = self.model.layer4[1](x0)
        else:
            x0 = self.model.layer4(x0)
        x0 = self.model.avgpool(x0)
        f0 = torch.squeeze(x0)
        x0 = self.classifier(f0)

        x1 = self.model1.layer3(x)
        if self.L4id[1] == 0:
            x1 = self.model1.layer4[0](x1)
        elif self.L4id[1] == 1:
            x1 = self.model1.layer4[0](x1)
            x1 = self.model1.layer4[1](x1)
        else:
            x1 = self.model1.layer4(x1)
        x1 = self.model1.avgpool(x1)
        f1 = torch.squeeze(x1)
        x1 = self.classifier1(f1)

        x2 = self.model2.layer3(x)
        if self.L4id[2] == 0:
            x2 = self.model2.layer4[0](x2)
        elif self.L4id[2] == 1:
            x2 = self.model2.layer4[0](x2)
            x2 = self.model2.layer4[1](x2)
        else:
            x2 = self.model2.layer4(x2)
        x2 = self.model2.avgpool(x2)
        f2 = torch.squeeze(x2)
        x2 = self.classifier2(f2)

        # x = self.model.layer3(x)
        #
        # if self.L4id[0] == 0:
        #     x0 = self.model.layer4[0](x)
        # elif self.L4id[0] == 1:
        #     x0 = self.model.layer4[0](x)
        #     x0 = self.model.layer4[1](x0)ßß
        # else:
        #     x0 = self.model.layer4(x)
        # x0 = self.model.avgpool(x0)
        # f0 = torch.squeeze(x0)
        # x0 = self.classifier(f0)
        #
        # if self.L4id[1] == 0:
        #     x1 = self.model1.layer4[0](x)
        # elif self.L4id[1] == 1:
        #     x1 = self.model1.layer4[0](x)
        #     x1 = self.model1.layer4[1](x1)
        # else:
        #     x1 = self.model1.layer4(x)
        # x1 = self.model1.avgpool(x1)
        # f1 = torch.squeeze(x1)
        # x1 = self.classifier1(f1)

        return f0, f1, f2, x0, x1, x2




