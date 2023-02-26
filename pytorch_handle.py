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
from torch.utils.tensorboard import SummaryWriter

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self,name):
        super().__init__() 
        self.label_file            = pd.read_csv(name)              
        self.encoder, self.decoder = self.return_encoder_decoder()
        self.img_labels            = self.get_img_labels()
        self.img_dir               = 'images_fb\\clean_images_256'        
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
        cat0_keys = self.label_file['cat:0'].unique()
        cat0_encoder = {}
        cat0_decoder = {}
        for _ in range(len(cat0_keys)):
            cat0_encoder[cat0_keys[_]] = _
            cat0_decoder[_] = cat0_keys[_]
        return cat0_encoder,cat0_decoder

def retrain_resnet_50(model,dataloaders,dataset_sizes,name,path):

    def train(model):
        epochs = 250
        optimizer =  torch.optim.SGD(model.parameters(), lr=0.025, momentum=0.875, weight_decay=3.0517578125e-05, nesterov=True)
 #       optimizer = torch.optim.Adam(model.parameters())       
     #  scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, step_size=5, gamma=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = int(10000/25))

        writer = SummaryWriter('logs')        
        best_accur = 0
        
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
                    writer.add_scalar('Loss/train', loss, i)
                    print(i)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()                                           
                    running_loss     += loss.item() * features.size(0)
                    running_corrects += torch.sum(preds == labels.data)                    

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc  = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                writer.add_scalar('Loss  vs epoch', epoch_loss, epoch)
                writer.add_scalar('Accur  vs epoch', epoch_acc, epoch)
                if epoch_acc >= best_accur:
                    best_accur = epoch_acc
                    save_model(model,os.path.join(path),f'epoch_{epoch},Loss_{epoch_loss:.4f} Acc_{epoch_acc:.4f}')

    ct = 0 
    for child in model.children():
        ct += 1 
        if ct >= 9: 
            for param in child.parameters(): 
                param.requires_grad = True

    path = os.path.join(path,name+str(datetime.now().strftime('%Y-%m-%d_%H_%M_%S')))
    device   = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model    = model.to(device)
    criterion= torch.nn.CrossEntropyLoss()    
    model    = train(model)

def save_model(model,path,name):   
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(),os.path.join(path,name+'.pt'))


def get_datasets():
    full_data= CustomImageDataset('training_data.csv')
    size    = len(full_data)
#    subset_sampler = torch.utils.data.SubsetRandomSampler(torch.randperm(size)[:500]) # Get split into test and training
 #   size =len(subset_sampler)
    train_size     = int(0.7 * size)
    val_size       = size -train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_data, [train_size,val_size])
    
    datasets    = {'train':train_dataset,
                    'val':val_dataset}
    
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=25, num_workers=4, shuffle=True)
                    for x in ['train', 'val']}
                    
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
    return dataloaders, dataset_sizes

def validate_test(model):


    full_data= CustomImageDataset('test.csv')
    dataloaders = torch.utils.data.DataLoader(full_data, batch_size=25, num_workers=4, shuffle=True)

    running_loss = 0.0

    running_corrects = 0

    device   = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model    = model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()    
    model.eval()   # Set model to evaluate mode
    
    for i,(features,labels) in enumerate(dataloaders):
        features = features.to(device)
        labels   = labels.to(device)
        with torch.no_grad():
            outputs  = model(features)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
        print(i)
        running_loss     += loss.item() * features.size(0)
        running_corrects += torch.sum(preds == labels.data)                    


    epoch_loss = running_loss / len(full_data)
    epoch_acc  = running_corrects.double() /len(full_data)

    print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')



if __name__ == '__main__':
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()
    torch.cuda.empty_cache()
 #   model    = models.resnet50( weights='IMAGENET1K_V2')#weights=("DEFAULT")pretrained=True
    model    = models.resnet50(  weights='IMAGENET1K_V2')
    model.fc = torch.nn.Linear(model.fc.in_features, 13)
    model.load_state_dict(torch.load('model_eval/deep_layers_True_start_epoch_11370002023-02-26_12_40_45/epoch_0,Loss_0.6061 Acc_0.8357.pt'))
    dataloaders,dataset_sizes = get_datasets()
    path   = 'model_eval' 
#    retrain_resnet_50(model,dataloaders,dataset_sizes,'_full_tr'+str(dataset_sizes['train']),path)
#    validate_test(model)
    pass
# %%
