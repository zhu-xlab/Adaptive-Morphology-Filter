import torch.nn.functional as F
import torchvision
import torch.nn as nn
import torch
import numpy as np
import cv2
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

expand = 1
class AD(nn.Module):
    def __init__(self, channel=1, ksize=3, expand=expand):
        super(AD, self).__init__()
        self.channel = channel
        self.ksize = ksize
        self.conv = nn.Conv2d(channel,9*expand*channel,3,1,padding=1,bias=False,groups=channel)  
        self.init_w()

    def init_w(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.fill_(0.)
                m.weight.data[:,:,1,1].fill_(1.)
        
    def forward(self, x):
        weight = self.conv.weight.data
        o,i,wh,ww = weight.shape
        weight = weight.view(o,i,wh*ww)
        self.conv.weight.data = F.gumbel_softmax(weight,hard=True).view(o,i,wh,ww)
        x = self.conv(x)
        b,c,h,w = x.shape
        x = x.view(b,self.channel,-1,h,w)
        x = torch.max(x,2)[0]
        return x

class AE(nn.Module):
    def __init__(self, channel=1, ksize=3, expand=expand):
        super(AE, self).__init__()
        self.channel = channel
        self.ksize = ksize
        self.conv = nn.Conv2d(channel,9*expand*channel,3,1,padding=1,bias=False,groups=channel)  
        self.init_w()

    def init_w(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.fill_(0.)
                m.weight.data[:,:,1,1].fill_(1.)

    def forward(self, x):
        weight = self.conv.weight.data
        o,i,wh,ww = weight.shape
        weight = weight.view(o,i,wh*ww)
        self.conv.weight.data = F.gumbel_softmax(weight,hard=True).view(o,i,wh,ww)
        x = self.conv(x)
        b,c,h,w = x.shape
        x = x.view(b,self.channel,-1,h,w)
        x = torch.min(x,2)[0]
        return x


class AEMP(nn.Module):
    def __init__(self, class_num=16, inp=4, order=5):
        super(AEMP, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(inp, 16, 7, 1, 7//2, bias=False),
                                  nn.BatchNorm2d(16),
                                  nn.LeakyReLU())
        channel=16

        self.AOs = nn.ModuleList([AE(channel,3) for i in range(order)])
        self.ACs = nn.ModuleList([AD(channel,3) for i in range(order)])

        self.fcs = nn.Sequential(
            nn.BatchNorm2d(channel*2*order+channel+inp),
            nn.Dropout(0.5),
            nn.Conv2d(channel*2*order+channel+inp, class_num, 1, 1, 0),
        )

    def maxpool(self,x):
        return F.max_pool2d(x,3,1,1)

    def minpool(self,x):
        return -F.max_pool2d(-x,3,1,1)

    def forward(self, inp):
        x = self.conv(inp)
        # x = inp
        aos = []
        temp = x
        for ao in self.AOs:
            temp = ao(temp)
            aos.append(temp)
            
        acs = []
        temp = x
        for ac in self.ACs:
            temp = ac(temp)
            acs.append(temp)

        feat = torch.cat(aos+[x]+acs+[inp],1)
        logist = self.fcs(feat)
        return logist

class AEMPLoss(nn.Module):
    def __init__(self,):
        super(AEMPLoss, self).__init__()   
    def forward(self, x, label, trainMask):
        label = torch.where(trainMask==1,label,-1*torch.ones_like(label).to(x.device))
        return F.cross_entropy(x, label, ignore_index=-1)
        
        
