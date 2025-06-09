import torch
import torch.nn as nn 
from torchvision import models


class FlashyFinetune(nn.Module):
    '''
    This is the vision encoder class for the 'flashiness' based model. 
    The idea was to fine-tune resnet50 to detect flashiness, then use
    this fine tuned model to cache image encodings and use those to train  
    a fusion based approach (model from fusion.py). 
    
    This encoder gave a MSLE of 5.33 (private score) (very dissapointing)
    '''
    
    def __init__(self, embedding_dim=64, image_dropout=0.3, regressor=None):
        super().__init__()
        self.features = [
            'image'
        ]
        self.targets = [
            'log_flashiness'
        ]
        self.embedding_dim = embedding_dim
       
        
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.vision_encoder = nn.Sequential(*list(self.resnet50.children())[:-1])
        self.image_dropout = nn.Dropout(image_dropout)
        if regressor is None: 
            regressor = nn.Sequential(
                nn.BatchNorm1d(2048),
                nn.Linear(2048,self.embedding_dim),
                nn.ReLU(),
                
                nn.BatchNorm1d(self.embedding_dim),
                nn.Dropout(0.2),
                nn.Linear(self.embedding_dim, 1)
            )
        self.regressor = regressor
            
        self.freeze_encoder()
        
    def freeze_encoder(self):
        for params in self.resnet50.parameters():
            params.requires_grad=False
    
    def unfreeze(self):
        '''
        Function to unfreeze the final layer of the vision encoder.
        '''
        for params in self.resnet50.layer4.parameters():
            params.requires_grad = True
            
        
    def forward(self, x):
        image = x['image']
        image_features = self.vision_encoder(image)
        image_features = torch.flatten(image_features, start_dim=1)
        image_features = self.image_dropout(image_features)
        output = self.regressor(image_features)
        return output.squeeze(1)
    
    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        
    def encode(self, x):
        with torch.no_grad():
            image = x['image']
            image_features = self.vision_encoder(image)
            image_features = torch.flatten(image_features, start_dim=1)
            return image_features
            
        
        
            
            
        
    