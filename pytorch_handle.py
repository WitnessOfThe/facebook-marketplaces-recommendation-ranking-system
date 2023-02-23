# %%
import os
import torch
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
from PIL import Image

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__() 
        self.label_file = pd.read_csv('training_data.csv')              
        self.encoder, self.decoder = self.return_encoder_decoder()
        self.img_labels = self.get_img_labels()
        self.img_dir    =  'images_fb\\clean_images'        
        self.x          = self.label_file['id_x']
        self.transform = transforms.Compose([transforms.ToTensor()])

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

def retrain_resnet_50():
    def train(model,epochs=10):
        model.train()
        phase = 'train'
        for epoch in range(epochs):
            for features,labels in dataloaders['train']:
                features = features.to(device)
                labels   = labels.to(device)
#                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(features)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    print(str(epoch)+str(loss))

    dataloaders = {x: torch.utils.data.DataLoader( CustomImageDataset(), batch_size=4,
                                             shuffle=True, num_workers=10)
                    for x in ['train', 'val']}
    device   = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#    model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features

    model.fc = torch.nn.Linear(num_ftrs, 13)
    model    = model.to(device)
    criterion= torch.nn.CrossEntropyLoss()    
    model    = train(model)

if __name__ == '__main__':
    retrain_resnet_50()
    pass
# %%
