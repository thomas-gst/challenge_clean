import torch
import torch.nn as nn 

class ImageRegressor(nn.Module):
    ''' 
    '''
    def __init__(self, embedding_dim = 128, image_dropout=0.4, residual=False, regressor=None):
        '''
        First model to use the video thumbnail. The approach is very naive, 
        just feeding the thumbnail with date features directly to an MLP regression head. 
        This model was an ablation test to see if the transformer was useful for combining features (it was). 
        
        This model gave an MSLE of 4.31 (private score).
        '''
        
        super().__init__()
        self.features = [
            'channel_mean_log_views',
            'year',
            'month_x',
            'month_y',
            'day_x',
            'day_y',
            'day_of_week_x',
            'day_of_week_y',
            'quarter_x',
            'quarter_y',
            'days_since_upload',
            'image_embedding',
        ]
        
        self.targets = ['log_views']
        self.residual = residual
        
        if regressor is None:
            regressor = nn.Sequential(
                nn.BatchNorm1d(768+11),
                nn.Dropout(0.3),
                nn.Linear(768+11, embedding_dim),
                nn.ReLU(),
                
                nn.BatchNorm1d(embedding_dim),
                nn.Dropout(0.1),
                nn.Linear(embedding_dim, 1)
            )
        self.regressor = regressor
        self.image_dropout = nn.Dropout(image_dropout)
        
    def forward(self, x):
        image_embedding = x['image_embedding']
        channel_mean_log_views = x['channel_mean_log_views'].unsqueeze(1)
        year = x['year'].unsqueeze(1)
        month_x = x['month_x'].unsqueeze(1)
        month_y = x['month_y'].unsqueeze(1)
        day_x = x['day_x'].unsqueeze(1)
        day_y = x['day_y'].unsqueeze(1)
        day_of_week_x = x['day_of_week_x'].unsqueeze(1)
        day_of_week_y = x['day_of_week_y'].unsqueeze(1)
        quarter_x = x['quarter_x'].unsqueeze(1)
        quarter_y = x['quarter_y'].unsqueeze(1)
        days_since_upload = x['days_since_upload'].unsqueeze(1)
        
        image_embedding = self.image_dropout(image_embedding)
        
        features = torch.cat([
            image_embedding,
            channel_mean_log_views,
            year,
            month_x,
            month_y,
            day_x,
            day_y,
            day_of_week_x,
            day_of_week_y,
            quarter_x,
            quarter_y,
            days_since_upload
        ], dim=1)
        
        regressor_ouput = self.regressor(features)
        if self.residual:
            output = regressor_ouput + channel_mean_log_views
        else:
            output = regressor_ouput
        return output.squeeze(1) 
    
    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)
        
    def save_model(self, path):
        torch.save(self.state_dict(), path)
        
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        

        