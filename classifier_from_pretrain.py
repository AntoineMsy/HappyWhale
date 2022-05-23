# -*- coding: utf-8 -*-
"""
Created on Fri May 13 14:59:07 2022

@author: antoi
"""
#from comet_ml import Experiment
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import os
import numpy as np
import pandas as pd
from utils import show
from resnet_td import ResNet18
from tqdm import tqdm
"""
def data_processing(data, data_type="train"):
""" 

class ConvNet_Classifier(nn.Module):
    def __init__(self, conv_net, output_size, num_classes):
        super().__init__()
        self.conv_net = conv_net
        self.linear = nn.Linear(output_size, num_classes)
    def forward(self, batch_data):
        x = self.conv_net(batch_data)
        x = self.linear(x)
        return x
classes = {'dusky_dolphin', 
           'long_finned_pilot_whale', 
           'short_finned_pilot_whale', 
           'spotted_dolphin', 
           'commersons_dolphin', 
           'common_dolphin', 
           'humpback_whale', 
           'fin_whale', 
           'bottlenose_dolphin', 
           'gray_whale', 
           'pygmy_killer_whale', 
           'cuviers_beaked_whale',  
           'killer_whale', 
           'false_killer_whale', 
           'blue_whale', 
           'brydes_whale', 
           'spinner_dolphin', 
           'beluga', 
           'southern_right_whale', 
           'melon_headed_whale', 
           'white_sided_dolphin', 
           'pantropic_spotted_dolphin', 
           'minke_whale'}

classes_map = """
            dusky_dolphin 0
           long_finned_pilot_whale 1
           short_finned_pilot_whale 2
           spotted_dolphin 3
           commersons_dolphin 4
           common_dolphin 5
           humpback_whale 6
           fin_whale 7
           bottlenose_dolphin 8
           gray_whale 9
           pygmy_killer_whale 10
           cuviers_beaked_whale 11
           killer_whale 12
           false_killer_whale 13
           blue_whale 14
           brydes_whale 15
           spinner_dolphin 16
           beluga 17
           southern_right_whale 18
           melon_headed_whale 19
           white_sided_dolphin 20
           pantropic_spotted_dolphin 21
           minke_whale 22
           sei_whale 23
           rough_toothed_dolphin 24
           frasiers_dolphin 25
           """
           
class LabelTransform:
    """Maps classes to integers and vice versa"""
    def __init__(self, classes_map_str):
        self.classes_map = {}
        self.index_map = {}
        for line in classes_map_str.strip().split('\n'):
            species, index = line.split()
            self.classes_map[species] = int(index)
            self.index_map[int(index)] = species
        self.index_map[1] = ' '

    def class_to_int(self, species):  
        return self.classes_map[species]

    def int_to_class(self, label):
        return self.index_map[label]
    
# Creating Custom Dataset for the project
class WhaleDataset(data.Dataset):
    def __init__(self, csv, image_folder, classes_map):
        
        self.csv = pd.read_csv(csv)
        self.image_folder = image_folder
        self.resize = transforms.Compose([
                                         transforms.ToPILImage(),
                                         transforms.Resize((224,224)),
                                         transforms.ToTensor()
        ])
     
    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, idx):
        img = torchvision.io.read_image(self.image_folder + self.csv["image"].iloc[idx]).float()
       
        if img.shape[0] == 1:
            img = torch.cat((img,img,img))
        
        img = self.resize(img)
       
        label = self.csv["species"].iloc[idx]
        return {"image" :img, "label": label}
    
class IterMeter(object):
    """keeps track of total iterations"""
    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val

def train(model, device, train_loader, optimizer, scheduler, epoch, iter_meter, experiment, criterion):
    model.train()
    print("starting training")
    data_len = len(train_loader.dataset)
    if experiment != None :
        with experiment.train():
            for batch_idx, _data in tqdm(enumerate(train_loader)):

                image = _data["image"] 
                label = _data["label"]

                image = image.to(device)
                label = label.to(device)
                optimizer.zero_grad()

                output = model(image)
                softmax = nn.Softmax(dim =1)
                output = softmax(output)

                # (batch, time, n_class)
                loss = criterion(output, label)
                loss.backward()


                optimizer.step()
                scheduler.step()
                iter_meter.step()
                experiment.log_metric('loss', loss.item(), step=iter_meter.get())
                experiment.log_metric('learning_rate', scheduler.get_lr(), step=iter_meter.get())
        
                optimizer.step()
                scheduler.step()
                iter_meter.step()
                """
                if batch_idx % 100 == 0 or batch_idx == data_len:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), data_len,
                        100. * batch_idx / len(train_loader), loss.item()))
                """
    else:
       
        for batch_idx, _data in tqdm(enumerate(train_loader)):

            image = _data["image"] 
            label = _data["label"]

            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()

            output = model(image)
            softmax = nn.Softmax(dim =1)
            output = softmax(output)

            # (batch, time, n_class)
            loss = criterion(output, label)
            loss.backward()


            optimizer.step()
            scheduler.step()
            iter_meter.step()
            """
            if batch_idx % 100 == 0 or batch_idx == data_len:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), data_len,
                    100. * batch_idx / len(train_loader), loss.item()))
            """

def test(model, device, test_loader, epoch, iter_meter, experiment, criterion):
    print('\nevaluating…')
    model.eval()
    test_loss = 0
    
    with experiment.test():
        with torch.no_grad():
            for I, _data in enumerate(test_loader):
               image, label = _data 
               image = image.to(device)
    
               output = model(image)  # (batch, time, n_class)
               
               loss = criterion(output, label)
               
               test_loss += loss.item() / len(test_loader)
               
    experiment.log_metric('test_loss', test_loss, step=iter_meter.get())
  
    print('Test set: Average loss: {:.4f}\n'.format(test_loss))


def main(model, experiment = None, learning_rate=5e-4, batch_size=256, epochs=5):
    
    hparams = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs
    }
    if experiment != None :
        experiment.log_parameters(hparams)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print(device)
    
    model.to(device)
    
    #print("saved")
    
   
    
    train_dataset = WhaleDataset("train_corrected.csv", "train_images/", classes_map)
    #test_dataset = WhaleDataset(csv, image_folder, classes_map)
    kwargs = {'num_workers': 6, 'pin_memory': True} if use_cuda else {}
    train_loader = data.DataLoader(dataset=train_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=True,
                                
                                **kwargs)
    """
    test_loader = data.DataLoader(dataset=test_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=False,
                                collate_fn=lambda x: data_processing(x, 'valid'),
                                **kwargs)
    """
    
    
    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 1000)

    
    iter_meter = IterMeter()
    
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, scheduler, epoch, iter_meter, experiment, criterion)
        torch.save(model.state_dict(), "resnet.pt")
        #test(model, device, train_loader, epoch, iter_meter, experiment)
   
    
if __name__ == "__main__":
    print("starting script")
    """
    experiment = Experiment(
    api_key="3R2GzsUplN6iNSQJFeYBO0gD4",
    project_name="modal",
    workspace="antoinemsy",
)
    """
    conv_model = torchvision.models.resnet18(pretrained = True)
    model = ConvNet_Classifier(conv_model,1000,27)
    
    print("experiment loaded")
    
    main(model,experiment)
    