from comet_ml import Experiment
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

class WhaleDataset_eval(data.Dataset):
    ## Dataset object, returning a triplet (img, auto augmented image, label) 
    def __init__(self, csv):
        self.df = pd.read_csv(csv)
        self.groups = self.df.groupby('individual_id').groups
        self.keys = list(self.groups.keys())
        self.label_encoder = {}
        self.augment = transforms.Compose([
                                        torchvision.transforms.AutoAugment()
                                       
                                         ])
        for i in range(len(self.keys)):
            self.label_encoder[self.keys[i]] = i
        
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
            "aug" : img_aug,
            'id' : label
        }

    
class Resnet_Classifier(nn.Module):
    #Resnet50 Classifier described in the report
    def __init__(self, in_size, out_size):
        super().__init__()
        self.conv_model = torchvision.models.resnet50(pretrained = True)
        self.classifier =  nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_size, in_size),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(in_size, in_size),
            nn.Tanh(),
            nn.Linear(in_size, out_size)
        )     
        self.linear_classifier = nn.Sequential(nn.Linear(in_size,out_size), nn.Softmax(dim=1))
        self.linear = nn.Linear(in_size,out_size)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, batch_data):
        x = self.conv_model(batch_data)
        x = self.classifier(x)
        return x

class IterMeter(object):
    """keeps track of total iterations"""
    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val
    
def train(model, train_loader, optimizer, criterion, scheduler, device, experiment, iter_meter):
    model.train()
    softmax = nn.Softmax(dim=1)
    with experiment.train():
        for batch_idx, _data in tqdm(enumerate(train_loader)):
                image = _data["image"]
                image_aug = _data["aug"]
                target = _data["id"]
                image, image_aug,target = image.to(device), image_aug.to(device), target.to(device)
       
                optimizer.zero_grad()
                
                pred = model(image)
                
                pred_aug = model(image_aug)
                
                
                loss = 2/3*criterion(pred,target) + 1/3*criterion(pred_aug,target)
           
                loss.backward()
                optimizer.step()
                scheduler.step()
                iter_meter.step()
                
                #Logging metrics on comet
                experiment.log_metric('loss', loss.item(), step=iter_meter.get())
                experiment.log_metric('learning_rate', scheduler.get_lr(), step=iter_meter.get())

def evaluate(model, dataset, criterion,device,batch_size, experiment, iter_meter):
    _, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) -5000,5000])
    kwargs = {'num_workers': 6, 'pin_memory': True} if use_cuda else {}
    test_loader= data.DataLoader(dataset=test_dataset,
                            batch_size=hparams['batch_size'],
                            shuffle=True,
                            **kwargs)
    avg_loss = 0
    l = len(test_loader)
    acc = 0
    model.eval()
    with experiment.test():
        with torch.no_grad():
            for batch_idx, _data in tqdm(enumerate(test_loader)):
                image = _data["image"]

                target = _data["id"]

                image, target = image.to(device), target.to(device)
                pred = model(image)
                loss = criterion(pred,target)
                avg_loss += loss/l

                #compute accuracy
                _, predicted_ids = torch.max(pred,1)
                acc += (predicted_ids == target).sum().item()/l/batch_size
                
    #logging metrics on comet
    experiment.log_metric('test_loss', loss, step=iter_meter.get())
    experiment.log_metric('accuracy', acc, step=iter_meter.get())
    print(acc)
    return acc

def main(model, dataset, experiment, learning_rate = 3e-4, batch_size = 60, epochs = 51):
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
    
    kwargs = {'num_workers': 6, 'pin_memory': True} if use_cuda else {}
    train_loader = data.DataLoader(dataset=dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=True,
                                **kwargs)
    
    optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = len(train_loader))
    iter_meter = IterMeter()
    
    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))
    print("it # per epoch : " + str(len(train_loader)))
    
    for i in range(1, epochs+1):
        print("Epoch # : " + str(i))
        train(model, train_loader, optimizer, criterion, scheduler, device, experiment, iter_meter)
        evaluate(model, dataset,criterion, device, batch_size, experiment,iter_meter)
        
    torch.save(model.state_dict(), "ResNetclassifier.pt")
    
if __name__ == "__main__":
   
    dataset = WhaleDataset_eval("train_corrected.csv")
    model = Resnet_Classifier(1000,len(dataset.keys))
    
    model.load_state_dict(torch.load("ResNetclassifier1.pt", map_location = torch.device('cuda'))) 
    experiment = Experiment(
    api_key="3R2GzsUplN6iNSQJFeYBO0gD4",
    project_name="modal",
    workspace="antoinemsy",
)
              
    main(model, dataset, experiment)