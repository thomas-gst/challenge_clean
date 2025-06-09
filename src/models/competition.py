import torch
import torch.nn as nn 
from transformers import BertConfig, BertModel

class CompetitionModel(nn.Module):
    '''
    Additional siamese model : a fusion transformer approach to predict, 
    given two different videos, which has the most views. In practice, 
    it was completely untrainable. 
    
    No submissions were made with this model. 
    '''
    def __init__(self, hidden_dim = 384, dropout=0):
        super().__init__()
        self.features = [
            'channel_int', 
            'year',
            'month',
            'day',
            'day_of_week',
            'quarter',
            'days_since_upload',
            'image_embedding',
            'title_embedding',
            'encoded_short',
            'channel_mean_log_views',
        ]
        
        
        self.image_projector = nn.Sequential(nn.Linear(768, hidden_dim), nn.Dropout(dropout))
        self.text_projector = nn.Sequential(nn.Linear(384, hidden_dim), nn.Dropout(dropout))
        self.numeric_projector = nn.Sequential(nn.Linear(7, hidden_dim), nn.Dropout(dropout)) # 6 date + 1 for channel mean log views
        self.channel_embedding = nn.Sequential(nn.Embedding(46, hidden_dim))
        self.short_embedding = nn.Sequential(nn.Embedding(2, hidden_dim))
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.task = 'competition'
        
        self.config = BertConfig(
            hidden_size=hidden_dim,
            num_attention_heads=32,
            num_hidden_layers=2,
            intermediate_size=hidden_dim * 1,
            hidden_dropout_prob=dropout,
            max_position_embeddings=6,
        )
        
        self.transformer = BertModel(self.config)
        
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(2*hidden_dim),
            nn.Linear(2*hidden_dim, 2),
            nn.GELU(),
            #nn.BatchNorm1d(32),
            #nn.Linear(32, 1),
        
        )
        
        self.softmax_layer = nn.Softmax(dim=1)
        
    def _build_tokens(self, x):
        image_features = x['encoded_image']
        title_features = x['encoded_title']
        encoded_channel = x['encoded_channel']
        encoded_short = x['encoded_short']
        channel_mean = x['channel_mean_log_views']
        year = x['year']
        month = x['month']
        day = x['day']
        day_of_week = x['day_of_week']
        quarter = x['quarter']
        days_since_upload = x['days_since_upload']
        
        numeric_inputs = torch.stack([year, month, day, day_of_week, quarter, days_since_upload, channel_mean], dim=1)
        numeric_features = self.numeric_projector(numeric_inputs)
        image_features = self.image_projector(image_features)
        text_features = self.text_projector(title_features)
        channel_emb = self.channel_embedding(encoded_channel)
        short_emb = self.short_embedding(encoded_short)
        
        cls_token = self.cls_token.expand(image_features.size(0), -1, -1)
        image_features = image_features.unsqueeze(1)
        text_features = text_features.unsqueeze(1)
        channel_emb = channel_emb.unsqueeze(1)
        short_emb = short_emb.unsqueeze(1)
        numeric_features = numeric_features.unsqueeze(1)
        
        tokens = torch.cat([cls_token, image_features, text_features, short_emb, channel_emb, numeric_features], dim=1)
        
        return tokens
    
    def forward(self, x_1, x_2):
        tokens_1 = self._build_tokens(x_1)
        tokens_2 = self._build_tokens(x_2)
        
        attention_mask = torch.ones((tokens_1.size(0), tokens_1.size(1)), device=tokens_1.device)
        attention_mask[:, 3] = 0 # nobody attends to short_embedding
        attention_mask[:, 4] = 0 # nobody attends to channel_emb
        attention_mask[:, 5] = 0 # nobody attends to numeric_features
        
        fused_1 = self.transformer(
            inputs_embeds=tokens_1,         
            attention_mask=attention_mask  
        ).last_hidden_state[:, 0]
        
        fused_2 = self.transformer(
            inputs_embeds=tokens_2,         
            attention_mask=attention_mask  
        ).last_hidden_state[:, 0]
        
        classifier_input = torch.cat([fused_1, fused_2], dim=1)
        classifier_output = self.classifier(classifier_input)

        
        return classifier_output
        
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)