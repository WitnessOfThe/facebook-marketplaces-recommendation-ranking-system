# %%
import os
import torch
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
from datetime import datetime
from PIL import Image
from datetime import datetime
import copy
from torch.utils.tensorboard import SummaryWriter

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__() 
        self.label_file            = pd.read_csv('training_with_reserved_test.csv')              
        self.encoder, self.decoder = self.return_encoder_decoder()
        self.img_labels            = self.get_img_labels()
        self.img_dir               =  'images_fb\\clean_images_256'        
        self.x                     = self.label_file['id_x']
        self.transform             = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.img_labels)

    def get_img_labels(self):
        filenames       = self.label_file['id_x']+'.jpg'
        encode_category = self.label_file['cat:0'].map(self.encoder)
        return pd.DataFrame({'name':filenames,'cat':encode_category})

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx,0])
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx,1]
        return self.transform(image), label

    def return_encoder_decoder(self):
        cat0_keys = self.label_file['cat:0'].unique()
        cat0_encoder = {}
        cat0_decoder = {}
        for _ in range(len(cat0_keys)):
            cat0_encoder[cat0_keys[_]] = _
            cat0_decoder[_] = cat0_keys[_]
        return cat0_encoder,cat0_decoder

def retrain_resnet_50(model,dataloaders,dataset_sizes,name,path,unfreeze2layers=False):
    def train(model):
        epochs = 100
        model.train()
        phase = 'train'
        optimizer        =  torch.optim.SGD(model.fc.parameters(), lr=0.01, momentum=0.9)
        exp_lr_scheduler =  torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
 #       best_model = copy.deepcopy(model.state_dict())+
        best_acc = 0.0

        writer = SummaryWriter('logs')        
#        weights=ResNet50_Weights.DEFAULT
        for epoch in range(epochs):
            for phase in ['train','val']:

                running_loss = 0.0
                running_corrects = 0
                
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode
                for i,(features,labels) in enumerate(dataloaders[phase]):
                    features = features.to(device)
                    labels   = labels.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs  = model(features)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
  #                      print(f'{phase} Loss: {loss:.4f} Acc: {torch.sum(preds == labels.data):.4f}')
                    writer.add_scalar('Loss/train', loss, i)
                    print(i)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()                                           
                    running_loss     += loss.item() * features.size(0)
                    running_corrects += torch.sum(preds == labels.data)                    

            if phase == 'train':
                exp_lr_scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc  = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            if phase == 'val':
                save_model(model,os.path.join(path),f'epoch_{epoch}, Loss_{epoch_loss:.4f} Acc_ {epoch_acc:.4f}')
     #           best_model = copy.deepcopy(model.state_dict())
    #    model.load_state_dict(best_model)
     #   return model
    if unfreeze2layers:
        for param in model.layer4.parameters():
            param.requires_grad = True

        for param in model.layer3.parameters():
            param.requires_grad = True
        name = 'deep_layers_True'+name

    path = os.path.join(path,name+str(datetime.now().strftime('%Y-%m-%d_%H_%M_%S')))
    save_model(model,path,'StartingModel')
    device   = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_ftrs = model.fc.in_features

    model.fc = torch.nn.Linear(num_ftrs, 13)
    model    = model.to(device)
    criterion= torch.nn.CrossEntropyLoss()    
    model    = train(model)

def save_model(model,path,name):   
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(),os.path.join(path,name+'.pt'))


def get_datasets():
    full_data= CustomImageDataset()
    size    = len(full_data)
#    subset_sampler = torch.utils.data.SubsetRandomSampler(torch.randperm(size)[:500]) # Get split into test and training
 #   size =len(subset_sampler)
    train_size     = int(0.7 * size)
    val_size       = int(0.2  * size)
    test_size      = size-train_size-val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_data, [train_size,val_size,test_size])
    
    datasets    = {'train':train_dataset,
                    'val':val_dataset,
                    'test':test_dataset}
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=25, num_workers=4, shuffle=True)
                    for x in ['train', 'val','test']}
                    
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
    return dataloaders,dataset_sizes

if __name__ == '__main__':
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()
    torch.cuda.empty_cache()
    model    = models.resnet50( weights=("DEFAULT"))#weights=("DEFAULT")pretrained=True
  #  model = torch.load('model_eval/pretrained_subset_size77852023-02-24_00_46_28/epoch_7_Loss_1.4963_Acc_0.5306.pt')
    dataloaders,dataset_sizes = get_datasets()
    path   = 'model_eval' 
    retrain_resnet_50(model,dataloaders,dataset_sizes,'dif_learning_rate_subset_size'+str(dataset_sizes['train']),path,True)
    pass
# %%
