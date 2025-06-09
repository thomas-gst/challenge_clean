import torch 
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class VideoDataset(Dataset):
    '''
    Custom Dataset class:
    The main paradigm was to limit memory overhead, while ensuring the 
    code was reusable. Hence, the dataset only keeps in memory the relevant 
    columns of the dataframe (as indicated by the current model's features and targets),
    and creates batch dictionnaries, ie the __get_item__ method returns a dictionary where
    each value is a tensor of size [batch, ...]. 
    '''
    def __init__(self, data_frame, model_features, model_targets, transform=None, image_path=None, test=False):
        self.features = model_features + model_targets + ['id']
        self.transform = transform
        self.image_path = image_path
        self.df = data_frame[[f for f in self.features if f != 'image']]
        self.test = test

    def __len__(self):
        return len(self.df)
    
    def _load_and_transform_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            return self.transform(image)
        else:
            return image
        
    def _load_int(self, value):
        return torch.tensor(value, dtype=torch.long)
    
    def _load_float(self, value):
        return torch.tensor(value, dtype=torch.float32)

    def _load_str(self, value):
        return value

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dict = {}
        for k in self.features:
            if k == 'channel_int':
                dict[k] = self._load_int(row.loc[k])
                continue
            elif k == 'short':
                dict[k] = self._load_int(row.loc[k])
                continue
            elif k == 'anomaly_class':
                if self.test:
                    dict[k] = torch.tensor(0, dtype=torch.long)
                    continue
                else:
                    dict[k] = self._load_int(row.loc[k])
                    continue
            elif k == 'log_views':
                if self.test:
                    dict[k] = torch.tensor(0.0, dtype=torch.float32)
                    continue
                else:
                    dict[k] = self._load_float(row.loc[k])
                    continue
            elif k == 'image':
                if self.test:
                    dict[k] = self._load_and_transform_image(f"{self.image_path}/{int(row.loc['id'])}.jpg")
                else:
                    dict[k] = self._load_and_transform_image(f"{self.image_path}/{row.loc['id']}.jpg")
                continue
            elif k == 'id':
                dict[k] = self._load_str(row.loc[k])
                continue
            else:
                dict[k] = self._load_float(row.loc[k])
        return dict
    
           
 
