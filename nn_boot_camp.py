# %%
import os
import torch
import pandas as pd
import pickle
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
from datetime import datetime
from PIL import Image
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self,name):
        super().__init__() 
        self.label_file            = pd.read_csv(name)              
        self.encoder, self.decoder = self.return_encoder_decoder()
        self.img_labels            = self.get_img_labels()
        self.img_dir               = 'images_fb\\clean_images_224'        
        self.x                     = self.label_file['id_x']
        self.transform             = transforms.Compose([transforms.RandomVerticalFlip(),
                                                         transforms.RandomRotation((-180,+180)),
                                                         transforms.RandomHorizontalFlip(),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

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
        with open(os.path.join('model_final','decoder.pkl'), 'rb') as f:
            decoder =  pickle.load(f)        
        with open(os.path.join('model_final','encoder.pkl'), 'rb') as f:
            encoder =  pickle.load(f)
        return encoder, decoder
#        cat0_keys = self.label_file['cat:0'].unique()
 #       cat0_encoder = {}
  #      cat0_decoder = {}
   #     for _ in range(len(cat0_keys)):
    #        cat0_encoder[cat0_keys[_]] = _
     #       cat0_decoder[_] = cat0_keys[_]
 #       return cat0_encoder,cat0_decoder
 
def retrain_resnet_50(model,dataloaders,dataset_sizes,name,path):
    model.fc = torch.nn.Linear(model.fc.in_features, 13)
    # Unfreeze last 2 layers
    ct = 1 
    for child in model.children():
        ct += 1 
        if ct >= 9: 
            for param in child.parameters(): 
                param.requires_grad = True
        else:
            for param in child.parameters(): 
                param.requires_grad = False

    path  = os.path.join(path,name+str(datetime.now().strftime('%Y-%m-%d_%H_%M_%S')))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device to work with
    writer = SummaryWriter()        

    model    = model.to(device)
    # set up optimiser and scheduler with respect to resnet recomentations (can be found in Git)
    # or simply use Adam :)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.875, weight_decay=3.0517578125e-05)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
#    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 10,eta_min= 1E-6, last_epoch=-1)
    epochs     = 7*5
#    optimizer  = torch.optim.Adam(model.parameters())       
    criterion  = torch.nn.CrossEntropyLoss()    
    best_accur = 0
    
    for epoch in range(epochs):
        for phase in ['train','val']:
            
            model,running_loss,running_corrects = train_eval_loop(model,phase,dataloaders,device,criterion,optimizer,writer)

            if phase == 'train':
                writer.add_scalar('LR vs epoch', scheduler.get_last_lr()[0], epoch)
                scheduler.step()
                pass

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc  = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                writer.add_scalar('Eval: Loss  vs epoch', epoch_loss, epoch) # logging
                writer.add_scalar('Eval: Accur  vs epoch', epoch_acc, epoch) # logging
                if epoch_acc >= best_accur:
                    best_accur = epoch_acc
                    save_model(model,os.path.join(path),f'epoch_{epoch},Loss_{epoch_loss:.4f} Acc_{epoch_acc:.4f}')
            else:
                writer.add_scalar('train: Loss  vs epoch', epoch_loss, epoch) # logging
                writer.add_scalar('train: Accur  vs epoch', epoch_acc, epoch) # logging

    phase = 'test'
    model,running_loss,running_corrects = train_eval_loop(model,phase,dataloaders,device,criterion,optimizer,writer)

    epoch_loss = running_loss / dataset_sizes[phase]
    epoch_acc  = running_corrects.double() / dataset_sizes[phase]

    writer.add_scalar('test_loss', epoch_loss, epoch) # logging
    writer.add_scalar('test_acur', epoch_acc, epoch) # logging

    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

def train_eval_loop(model,phase,dataloaders,device,criterion,optimizer,writer):
    running_loss     = 0.0
    running_corrects = 0
    if phase == 'train':
        model.train()  # Set model to training mode
    else:
        model.eval()   # Set model to evaluate mode
    for i,(features,labels) in enumerate(dataloaders[phase]):
        features = features.to(device)
        labels   = labels.to(device)
        if phase == 'train':
            optimizer.zero_grad()
        with torch.set_grad_enabled(phase == 'train'):
            outputs  = model(features)
            _, preds = torch.max(outputs, 1)
            loss     = criterion(outputs, labels)
        print( i )
        if phase == 'train':
            loss.backward()
            optimizer.step()                                            
        running_loss     += loss.item() * features.size(0)
        running_corrects += torch.sum(preds == labels.data)                    
    return model,running_loss,running_corrects

def save_model(model,path,name):   
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(),os.path.join(path,name+'.pt'))

def get_datasets(data_set_path_name,data_test_set_path_name):
    full_data   = CustomImageDataset(data_set_path_name)
    test_dataset= CustomImageDataset(data_test_set_path_name)

    size           = len(full_data)
    train_size     = int(0.7*size)
    eval_size      = size -train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_data, [train_size,eval_size])

#    size           = len(full_data)
 #   train_size     = int(0.9*size)
  #  test_size      = size -train_size
   # train_dataset, val_dataset = torch.utils.data.random_split(full_data, [train_size,eval_size])

#    size           = len(full_data)
 #   train_size     = int(0.7*size)
  #  eval_size      = size -train_size
   # train_dataset, val_dataset = torch.utils.data.random_split(full_data, [train_size,eval_size])
    

    datasets    = {'train':train_dataset,
                    'val':val_dataset,
                    'test':test_dataset}
    
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=100, num_workers=1,shuffle=True)
                    for x in ['train', 'val','test']}
                    
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val','test']}
    return dataloaders, dataset_sizes

def validate_test(model):

    dataloaders,dataset_sizes = get_datasets('training_data_sandbox\\training_data_rm_dup.csv','training_data_sandbox\\test_data_rm_dup.csv')

    device   = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model    = model.to(device)
    model.eval()   
    criterion = torch.nn.CrossEntropyLoss()    

    phase = 'test'
    model,running_loss,running_corrects = train_eval_loop(model,phase,dataloaders,device,criterion,[],[])

    epoch_loss = running_loss / dataset_sizes[phase]
    epoch_acc  = running_corrects.double() / dataset_sizes[phase]

    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')    

if __name__ == '__main__':
    torch.manual_seed(18)
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()
    torch.cuda.empty_cache()

    """Validation of the trained Model"""

  #  model    = models.resnet50(  weights='IMAGENET1K_V2')#
   # model.fc = torch.nn.Linear(model.fc.in_features, 13)
#    model.load_state_dict(torch.load(os.path.join('model_eval','CosFullFarsh38492023-03-02_01_31_30','epoch_4,Loss_2.0318 Acc_0.3452.pt')))
   # validate_test(model)

    '''Validation of the Resnet'''
#    model    = models.resnet50(weights='IMAGENET1K_V2')
 #   model.fc = torch.nn.Linear(model.fc.in_features, 13)
  #  validate_test(model)

    '''Training of the model'''
    model    = models.resnet50(  weights='IMAGENET1K_V2')
    dataloaders,dataset_sizes = get_datasets('training_data_sandbox\\training_data_rm_dup.csv','training_data_sandbox\\test_data_rm_dup.csv')
    path   = 'model_eval' 
    retrain_resnet_50(model,dataloaders,dataset_sizes,'CosFullFarsh'+str(dataset_sizes['train']),path)
# %%
