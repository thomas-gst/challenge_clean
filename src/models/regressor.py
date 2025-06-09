import torch
import torch.nn as nn



class StandardRegressor(nn.Module):
    '''
    Basic regression based of the channel's mean log views and date features. 
    The idea was that the date was supposed to enrich the very basic guess based off of 
    the channel's mean log views by accounting for temporal evolution (eg a video from 2014 
    has been around longer so has probably more views than the channel average).
    
    This model gave an MSLE of 4.69 (private score)
    '''
    def __init__(self, embedding_dim=8):
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
            'days_since_upload'
        ]
        self.targets = ['log_views']
        self.embedding_dim = embedding_dim
        
       
        self.regressor = nn.Sequential(
            nn.Linear(11, embedding_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_dim),
            nn.Dropout(0.1),
            
            nn.Linear(embedding_dim, 1),
        )
        
    def forward(self, x):
        channel_mean_log_views = x['channel_mean_log_views']
        year = x['year']
        month_x = x['month_x']
        month_y = x['month_y']
        day_x = x['day_x']
        day_y = x['day_y']
        day_of_week_x = x['day_of_week_x']
        day_of_week_y = x['day_of_week_y']
        quarter_x = x['quarter_x']
        quarter_y = x['quarter_y']
        days_since_upload = x['days_since_upload']
    
        numeric_features = torch.stack([
            channel_mean_log_views, year, month_x, month_y, day_x, day_y,
            day_of_week_x, day_of_week_y, quarter_x, quarter_y, days_since_upload
        ], dim=1)

        return self.regressor(numeric_features).squeeze(1) + channel_mean_log_views
    
    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)
        
    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

