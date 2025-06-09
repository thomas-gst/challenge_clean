import torch.nn as nn 
import torch
from transformers import BertConfig, BertModel

class MultiTaskFusion(nn.Module):
    '''
    Main model : Fusion transformer combining title, channel, short, and 
    thumbnail to make a residual prediction based off of two different heads : a classifier that 
    aims to predict the anomaly class, and a regressor that aims to predict the residual noise.
    
    This model gave a MSLE of 3.72 (private score)
    '''
    def __init__(self, embedding_dim=64, token_dropout=0.35, transformer_config=None, classifier=None, classifier_dropout=0.3, regressor = None, regressor_droupout=0.4, image_embedding_dim=768):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.features = [
            'channel_int', 
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
            'title_embedding',
            'short',  
        ]
        
        self.classes = torch.tensor([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6])
        self.targets = ['log_views', 'anomaly_class', 'encoded_anomaly_class']
        
        self.vision_projector = nn.Sequential(nn.Linear(image_embedding_dim, embedding_dim), nn.LayerNorm(embedding_dim), nn.Dropout(token_dropout))
        self.text_projector = nn.Sequential(nn.Linear(384, embedding_dim), nn.LayerNorm(embedding_dim), nn.Dropout(token_dropout))
        self.short_embedding = nn.Sequential(nn.Embedding(2, embedding_dim), nn.LayerNorm(embedding_dim), nn.Dropout(token_dropout))
        self.channel_embedding = nn.Sequential(nn.Embedding(46, embedding_dim), nn.LayerNorm(embedding_dim), nn.Dropout(token_dropout))
        
        self.cls_token = nn.Parameter(torch.randn(1,1, embedding_dim))
        self.register_buffer('attention_mask', self._build_attention_mask())
        self.register_buffer('token_type_ids', self._build_token_type_ids())
        
        if transformer_config is None:
            transformer_config = BertConfig(
                hidden_size=embedding_dim,
                num_hidden_layers=1,
                num_attention_heads=4,
                intermediate_size=embedding_dim * 1,
                hidden_dropout_prob=0.4,
                type_vocab_size=5, # cls + image + text + channel + short
                max_position_embeddings=5,
            )
            
        self.transformer = BertModel(transformer_config)
        
        if classifier is None:
            classifier = nn.Sequential(
                nn.BatchNorm1d(embedding_dim),
                nn.Dropout(classifier_dropout+0.2),
                nn.Linear(embedding_dim, 16),
                nn.ReLU(),
                
                nn.BatchNorm1d(16),
                nn.Dropout(classifier_dropout+0.2),
                nn.Linear(16, 16),
                nn.ReLU(),
                
                nn.BatchNorm1d(16),
                nn.Dropout(classifier_dropout+0.1),
                nn.Linear(16, 16),
                nn.ReLU(),
                
                nn.BatchNorm1d(16),
                nn.Dropout(classifier_dropout-0.1),
                nn.Linear(16, 12),
            )
            
        self.classifier = classifier
        self.soft_max = nn.Softmax(dim=1)
        
        if regressor is None:
            regressor = nn.Sequential(
                nn.BatchNorm1d(embedding_dim+11),
                nn.Dropout(regressor_droupout+0.1),
                nn.Linear(embedding_dim+11, embedding_dim+11),
                nn.ReLU(),
                
                nn.BatchNorm1d(embedding_dim+11),
                nn.Dropout(regressor_droupout),
                nn.Linear(embedding_dim+11, 1),
                
            )
            
        self.regressor = regressor
        
    def _build_attention_mask(self):
        '''
        Helper function to build the attention mask according to the batch size. 
        The mask prevents channel and short embeddings from being enriched.
        '''
        attention_mask = torch.ones(1, 5, dtype=torch.long)
        attention_mask[:,3] = 0
        attention_mask[:,4] = 0
        return attention_mask
    
    def _build_token_type_ids(self):
        '''
        Helper function to build the token type ids according to the batch size. 
        The token type ids are used to distinguish between different types of tokens (cls, image, title, short, channel).
        It's essentially like positional encoding, but it encodes token types instead of positions, even though here they end being the same thing. (The order of tokens is always cls, image, title, channel, short).
        CLS token at index 0 will have type id 0 by default. 
        '''
        token_type_ids = torch.zeros(1, 5, dtype=torch.long)
        token_type_ids[:, 1] = 1
        token_type_ids[:, 2] = 2
        token_type_ids[:, 3] = 3
        token_type_ids[:, 4] = 4
        return token_type_ids
    
    def forward(self, x):
        image_embedding = x['image_embedding']
        title_embedding = x['title_embedding']
        short = x['short']
        channel_int = x['channel_int']
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
        
        batch_size = image_embedding.size(0)
        
        numeric_features = torch.stack([
            channel_mean_log_views, year, month_x, month_y, day_x, day_y,
            day_of_week_x, day_of_week_y, quarter_x, quarter_y, days_since_upload
        ], dim=1)
        image_features = self.vision_projector(image_embedding).unsqueeze(1)  
        text_features = self.text_projector(title_embedding).unsqueeze(1)  
        short_features = self.short_embedding(short).unsqueeze(1)   
        channel_features = self.channel_embedding(channel_int).unsqueeze(1) 
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        
        tokens = torch.cat([cls_token, image_features, text_features, channel_features, short_features], dim=1)
        
        attention_mask = self.attention_mask.expand(batch_size, -1)
        token_type_ids = self.token_type_ids.expand(batch_size, -1)
        
        fused = self.transformer(
            inputs_embeds = tokens, 
            attention_mask = attention_mask,
            token_type_ids = token_type_ids, 
        ).last_hidden_state[:,0]
        
        regressor_input = torch.cat([fused, numeric_features], dim=1)
        
        classifier_output = self.classifier(fused)
        regressor_output = self.regressor(regressor_input).squeeze(1)
        
        classifier_probabilites = self.soft_max(classifier_output)
        classes = self.classes.to(classifier_probabilites.device)
        classifier_prediction = torch.sum(classifier_probabilites * classes, dim=1)
        
        prediction = regressor_output # + classifier_prediction + channel_mean_log_views // test to see the usefulness of the residual connection
        
        return prediction, classifier_output
    
    def predict(self, x):
        with torch.no_grad():
            prediction,_ = self.forward(x)
        return prediction
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
        
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        
        
        