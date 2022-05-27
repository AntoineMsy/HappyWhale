# -*- coding: utf-8 -*-
"""
Created on Tue May 24 11:41:27 2022

@author: antoi
"""
from comet_ml import Experiment
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import os
import numpy as np
import pandas as pd

from tqdm import tqdm

class WhaleDataset_contrastive(data.Dataset):
    #outputs a pair of augmented images and standard images per image in the dataset
    def __init__(self, csv):
        
        self.df = pd.read_csv(csv)
        self.groups = self.df.groupby('individual_id').groups
        self.keys = list(self.groups.keys())
        self.label_encoder = {}
        for i in range(len(self.keys)):
            self.label_encoder[self.keys[i]] = i
        
        
        self.augment = transforms.Compose([
                                        torchvision.transforms.AutoAugment()
                                       
                                         ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):         
        img = torchvision.io.read_image("train_images_cropped/" + self.df["image"].iloc[index]).float()
       
        if img.shape[0] == 1:
            img = torch.cat((img,img,img))
        
        
        img_aug = self.augment(img.to(torch.uint8)).float()
        label = self.label_encoder[self.df["individual_id"].iloc[index]]
      
        return {
            'image': img,
            'image_aug' : img_aug,
            'id' : label
        }

def contrastive_loss(h1,h2,target, margin =100):
    #Contrastive loss used in a previous version of the algorithm involving siamese networks
    d = torch.nn.functional.pairwise_distance(h1, h2)
    loss = torch.mean(target*torch.square(d) + (1-target)*torch.square(torch.clamp(margin-d,min = 0,max = None)))
    return loss

def SupCon_loss(z,z_a,label, device, temperature = 0.1):
    #Computes the SupCon loss over the batch : see -> https://arxiv.org/abs/2004.11362
    loss = torch.zeros((1),requires_grad = True, device = device)
    
    loss = loss.clone()
    for i in range(len(label)):
        
        z_i = z[i]
        
        z_2i = z_a[i]
        label_i = label[i]
        Ai = 0
        A2i = 0
        Pi = []
        #Computes sum over A(i) and computes positive indices
        for j in range(len(label)):
            if j!=i:
                z_j = z[j]
                z_2j = z_a[j]
                label_j = label[j]
                
                Ai += torch.exp(torch.dot(z_j,z_i)/temperature) + torch.exp(torch.dot(z_2j,z_i)/temperature)
                A2i += torch.exp(torch.dot(z_j,z_2i)/temperature) + torch.exp(torch.dot(z_2j,z_2i)/temperature)
                
                if label_j.item() == label_i.item():
                    Pi.append(j)
                    
        Ai += torch.exp(torch.dot(z_i,z_2i)/temperature)
        
        A2i += torch.exp(torch.dot(z_i,z_2i)/temperature)
        
        #computes sum over p in Pi
        
        d = len(Pi)*2 +2 #accounts for the augmentation to get the "real" cardinal of Pi
        
        for p in Pi:
            z_p = z[p]
            z_2p = z_a[p]
            
            loss += -1/d*(torch.log(torch.exp(torch.dot(z_i,z_p)/temperature)/Ai) 
                          + torch.log(torch.exp(torch.dot(z_i,z_2p)/temperature)/Ai) 
                          + torch.log(torch.exp(torch.dot(z_2i,z_p)/temperature)/A2i) 
                          + torch.log(torch.exp(torch.dot(z_2i,z_2p)/temperature)/A2i))
            
    loss += -1/d*(torch.log(torch.exp(torch.dot(z_i,z_2i)/temperature)/Ai) 
                  + torch.log(torch.exp(torch.dot(z_i,z_2i)/temperature)/A2i) 
                  )

    """
    print(-1/d*(torch.log(torch.exp(torch.dot(z_i,z_2i)/temperature)/Ai) 
                  + torch.log(torch.exp(torch.dot(z_i,z_2i)/temperature)/A2i) 
                  ))
    """
  
    return loss

class Embedding_Network(nn.Module):
    def __init__(self, conv_model, out_size = 128):
        super().__init__()
        self.conv_net = conv_model
        self.linear = nn.Linear(1000, out_size)
    
    def forward(self, image):
        #Encoder
        large_rep = self.conv_net(image)
        #Normalization
        large_rep_n = nn.functional.normalize(large_rep,dim = 1)
        #Projection Network
        proj_rep = self.linear(large_rep_n)
        #Normalization
        proj_rep_n = nn.functional.normalize(proj_rep,dim = 1)
     
        return proj_rep_n
        
class IterMeter(object):
    """keeps track of total iterations"""
    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val

def train(model, device, train_loader, optimizer, epoch, iter_meter, scheduler, experiment):
    model.train()
    data_len = len(train_loader.dataset)
    if experiment != None :
        with experiment.train():
            print("in exp")
            for batch_idx, _data in tqdm(enumerate(train_loader)):
            
                image, aug_image, label = _data["image"], _data["image_aug"], _data["id"]
                image, aug_image, label = image.to(device), aug_image.to(device), label.to(device)

                optimizer.zero_grad()


                z, z_a = model(image), model(aug_image)

                loss = SupCon_loss(z,z_a,label,device)

                loss.backward()


                optimizer.step()
                scheduler.step()
                iter_meter.step()
                experiment.log_metric('loss', loss.item(), step=iter_meter.get())
                experiment.log_metric('learning_rate', scheduler.get_lr(), step=iter_meter.get())

    else:
        for batch_idx, _data in tqdm(enumerate(train_loader)):
            
            image, aug_image, label = _data["image"], _data["image_aug"], _data["id"]
            image, aug_image, label = image.to(device), aug_image.to(device), label.to(device)
            
            optimizer.zero_grad()

            
            z, z_a = model(image), model(aug_image)
         
            loss = SupCon_loss(z,z_a,label,device)
            
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
    
def test(model, device, dataset, epoch, iter_meter, experiment, batch_size = 32):
    print('\nevaluating…')
    model.eval()
    test_loss = 0
    _ , test_dataset = data.random_split(dataset,[0.8*len(dataset),1-0.8*len(dataset)])
    test_loader = data.DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                )
    with experiment.test():
        with torch.no_grad():
            for I, _data in enumerate(test_loader):
               image, label = _data 
               image = image.to(device)
    
               output = model(image)  # (batch, time, n_class)
               
               loss = contrastive_loss(output, label)
               
               test_loss += loss.item() / len(test_loader)
               
    experiment.log_metric('test_loss', test_loss, step=iter_meter.get())
  
    print('Test set: Average loss: {:.4f}\n'.format(test_loss))

def main(model, experiment = None, learning_rate=3e-4, batch_size=224, epochs=8):
    if experiment != None:
        print("loading comet experiment ...")
        
    hparams = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs
    }
    
    if experiment != None :
        experiment.log_parameters(hparams)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model.to(device)
    dataset = WhaleDataset_contrastive("train_corrected.csv")
    
    #test_dataset = WhaleDataset(csv, image_folder, classes_map)
    kwargs = {'num_workers': 6, 'pin_memory': True} if use_cuda else {}
    train_loader = data.DataLoader(dataset=dataset,
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
    _, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) -5000,5000])
    
    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))
    
    optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 100)

    iter_meter = IterMeter()
    
    print("it# per epochs : " + str(len(dataset)/batch_size))
    for epoch in range(1, epochs + 1):
        print(epoch)
        train(model, device, train_loader, optimizer, epoch, iter_meter, scheduler,experiment)
    torch.save(model.state_dict(), "res_net_supcon.pt")
        #test(model, device, test_loader, epoch, iter_meter, experiment)
   
    
if __name__ == "__main__":
    print("starting script")
    
    conv_model = torchvision.models.resnet18(pretrained = True)
    model = Embedding_Network(conv_model)
    
    
    experiment = Experiment(
    api_key="3R2GzsUplN6iNSQJFeYBO0gD4",
    project_name="modal",
    workspace="antoinemsy",
)

    main(model,experiment)
    
