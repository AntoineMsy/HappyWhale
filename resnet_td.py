# -*- coding: utf-8 -*-
"""
Created on Fri May 13 16:17:26 2022

@author: antoi
"""


import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        """
        in_planes : nombre de features en entrée
        planes : nombre de features en sortie (et en entrée de toutes
                les convolutions sauf la première)
        stride : stride de la première convolution
        """
        super().__init__()
        self.in_planes,self.planes = in_planes,planes

        self.conv1=nn.Conv2d(in_planes,planes,kernel_size=3,stride=stride,padding=1)

        self.conv=nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1)

        self.activation=nn.ReLU(inplace=True)

        self.norm=nn.BatchNorm2d(planes)

        self.shortcut=nn.Conv2d(in_planes,planes,kernel_size=1,stride=2,padding=0)

    def forward(self, x):
    
        x_initial=x
        x=self.conv1(x)
        x=self.norm(x)
        x=self.activation(x)
        x=self.conv(x)
        x=self.norm(x)

        if (self.in_planes!=self.planes):
            x_initial=self.shortcut(x_initial)

        x=self.activation(x+x_initial)

        return x

class ResNet(nn.Module):
    def __init__(self, Block, num_blocks, num_classes=26, num_filters=16, input_dim=3,input_size=224):
        """
        block : la classe de bloc résiduel à utiliser
        num_blocks : liste d'entiers correspondant à la taille des piles : [2,2] 
          correspond à deux piles contenant chacune deux fois Block. 
        num_classes : nombre de classes en sortie
        num_filters : nombre de features après la toute première convolution
        input_dim : nombre de features des données en entrée, par défaut 3 (les
          canaux RGB)
        input_size : image de taille input_size x input_size en entrée, sert à 
          calculer la taille de la couche FC
        """
        super().__init__()
    
        self.conv1 = nn.Conv2d(input_dim, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.n=len(num_blocks)
        self.planes=num_filters
    
        # N'oubliez pas de calculer out_size, la taille en entrée de la couche FC
        # en fonction de input_size, des autres données en entrée et du reste de
        # l'architecture
        
        pool_n=2
        self.pool = nn.AvgPool2d(pool_n)
        out_size=(num_filters*(2**self.n))*(np.floor(np.ceil(input_size/((2**self.n)))/pool_n)**2)
        self.pile=nn.ModuleList()
    
        for i in range(self.n):
            self.planes=self.planes*2
            self.pile.append(nn.Sequential(ResNetBlock(int(self.planes/2),self.planes,stride=2)))
            for j in range(1,num_blocks[i]):
                self.pile.append(nn.Sequential(ResNetBlock(self.planes,self.planes,stride=1)))
    
        self.linear = nn.Linear(int(out_size), num_classes)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        
        for i in range(len(self.pile)) :
            out = self.pile[i](out)
        
        
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
        


# (1 + 2*(1 + 1) + 2*(1 + 1) + 2*(1 + 1) + 2*(1 + 1)) + 1 = 18
def ResNet18(input_size=224):
    return ResNet(ResNetBlock, [2,2,2,2])

        