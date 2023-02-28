import uvicorn
import faiss
import pickle
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from fastapi import File
from fastapi import UploadFile
from fastapi import Form
import torch
import torch.nn as nn
from  torchvision import models
from pydantic import BaseModel
from image_processor import ImagePrep

class FeatureExtractor(nn.Module):
    def __init__(self,
                 decoder: dict = None):
        super(FeatureExtractor, self).__init__()
        self.model    = models.resnet50(  weights='IMAGENET1K_V2')
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 13)
        self.model.eval()
        
        self.decoder = decoder
        self.decoder = decoder

    def forward(self, image):
        x = self.model(image)
        return x
    def predict(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return x

# Don't change this, it will be useful for one of the methods in the API
class TextItem(BaseModel):
    text: str



try:
    feature_extr = FeatureExtractor()
    feature_extr.model.load_state_dict(torch.load(os.path.join('model_final','model_final.pt' ),map_location=torch.device('cpu')))
    feature_extr.model.eval()

    with open(os.path.join('model_final','decoder.pkl'), 'rb') as f:
        decoder =  pickle.load(f)        
    with open(os.path.join('model_final','encoder.pkl'), 'rb') as f:
        incoder =  pickle.load(f)
except:
    raise OSError("No Feature Extraction model found. Check that you have the decoder and the model in the correct location")

try:
    index = faiss.read_index("FAISS_index.pkl")
    pass
except:
    raise OSError("No Image model found. Check that you have the encoder and the model in the correct location")


app = FastAPI()
print("Starting server")

@app.get('/healthcheck')
def healthcheck():
  msg = "API is up and running!"
  return {"message": msg}

  
@app.post('/predict/feature_embedding')
async def predict_image(file: UploadFile = File(...)):
    contents = file.file.read()
    with open(file.filename, 'wb') as f:
        f.write(contents) 
    with open(file.filename, 'rb') as f:
        pil_image = Image.open(file.filename)

    ip  = ImagePrep(pil_image)
    tens_image = ip.img
    img_emb = feature_extr(tens_image)
    return JSONResponse(content={
                                "features": img_emb.tolist()[0], # Return the image embeddings here   
                                    })
  
@app.post('/predict/similar_images')
async def predict_combined(file: UploadFile = File(...)):
    contents = file.file.read()
    with open(file.filename, 'wb') as f:
        f.write(contents) 
    with open(file.filename, 'rb') as f:
        pil_image = Image.open(file.filename)

    tens_image = ImagePrep(pil_image).img
    img_emb = feature_extr(tens_image)
    _, I = index.search(img_emb.detach().numpy(), 5)     # actual search

    return JSONResponse(content={
    "similar_index": I.tolist()[0], # Return the index of similar images here
        })
    
if __name__ == '__main__':
  uvicorn.run("api:app", host="0.0.0.0", port=8080)