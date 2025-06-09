import torchvision.io as io
import os 
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import pandas as pd
from torchvision import transforms as T
import numpy as np
from datetime import timezone
from tqdm import tqdm
from PIL import Image
from transformers import AutoModel
import torch
import faiss 


class Enricher:
    '''
    Custom class for feature engineering :
    The main paradigm is to prevent re-calculations, hence this class aims to 
    cache as many features as possible, storing them in .parquet files, greatly 
    speeding up training. 
    '''
    def __init__(self, train_val_image_path='data/raw/train_val', test_image_path='data/raw/test/', train_val_csv='data/raw/train_val.csv', test_csv='data/raw/test.csv', output_file='data/processed/'):
        self.output_file = output_file
        self.train_val_image_path = train_val_image_path
        self.test_image_path = test_image_path
        self.train_val_csv = train_val_csv
        self.test_csv = test_csv
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModel.from_pretrained("facebook/dinov2-base").to(self.device).eval()
        self.image_transform = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.label_encoder = LabelEncoder()
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.IndexFlatL2(384)
        
    def _add_date_features(self, df):
        '''Helper function to add date features to the DataFrame. Uses a sine-cosine transformation for cyclical features.'''
        if 'days_since_upload' not in df.columns:
            print("Creating date features...")
            df['date'] = pd.to_datetime(df['date'])
            df['year'] = df['date'].dt.year
            df['month_x'] = np.sin(df['date'].dt.month / 12 * 2 * np.pi)
            df['month_y'] = np.cos(df['date'].dt.month / 12 * 2 * np.pi)
            df['day_x'] = np.sin(df['date'].dt.day / 31 * 2 * np.pi)
            df['day_y'] = np.cos(df['date'].dt.day / 31 * 2 * np.pi)
            df['day_of_week_x'] = np.sin(df['date'].dt.dayofweek / 7 * 2 * np.pi)
            df['day_of_week_y'] = np.cos(df['date'].dt.dayofweek / 7 * 2 * np.pi)
            df['quarter_x'] = np.sin(df['date'].dt.quarter / 4 * 2 * np.pi)
            df['quarter_y'] = np.cos(df['date'].dt.quarter / 4 * 2 * np.pi)
            if df['date'].dt.tz is not None:
                reference_date = pd.Timestamp.now(timezone.utc)
            else:
                reference_date = pd.Timestamp.now().tz_localize(None)
            df['days_since_upload'] = (reference_date - df['date']).dt.days
        return df
    
    def _add_channel_int(self, df, test=False):
        '''
        Helper function to encode the 'channel' column using LabelEncoder. Each channel is transformed into a unique integer.
        When test=True, it uses the label encoder fitted on the training data to ensure consistent encoding.
        '''
        if test and 'channel_int' not in df.columns:
            print("Adding channel integer label as fitted while enriching train dataset ...")
            df['channel_int'] = self.label_encoder.transform(df['channel'])
        elif 'channel_int' not in df.columns:
            print("Adding channel integer label...")
            df['channel_int'] = self.label_encoder.fit_transform(df['channel'])
        return df
    
    def _add_title_embedding(self, df):
        '''Helper function to add title embeddings using SentenceTransformer. The output is a 384-dimensional vector for each title.'''
        if 'title_embedding' not in df.columns:
            print("Adding title embedding...")
            title_embeddings = self.sentence_transformer.encode(df['title'].tolist(), show_progress_bar=True, device=self.device)
            df['title_embedding'] = list(title_embeddings)
        return df
    
    def _add_image_embedding(self, df, image_path):
        '''Helper function to add image embeddings using DINOv2. the output is a 768-dimensional vector for each image.'''
        if 'image_embedding' not in df.columns:
            print("Adding image embedding...")
            image_embeddings = []
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Encoding images"):
                image_file = os.path.join(image_path, f"{row['id']}.jpg")
                if os.path.exists(image_file):
                    image = Image.open(image_file).convert('RGB')
                    image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        embedding = self.model(image_tensor).last_hidden_state.mean(dim=1).cpu().numpy()
                    image_embeddings.append(embedding[0])
                else:
                    print(f"Image file {image_file} not found. Appending zero vector.")
                    image_embeddings.append(np.zeros((768,)))
            df['image_embedding'] = list(image_embeddings)
        return df
                    
    def _add_log_views(self, df, test=False):
        '''Helper function to add log views.'''
        if test and 'log_views' not in df.columns:
            print("Adding log views as nan...")
            df['log_views'] = np.nan
        elif 'log_views' not in df.columns:
            print("Adding log views...")
            df['log_views'] = np.log1p(df['views'])
        return df
    
    def _get_channel_mean_log_views(self, df):
        '''Helper function to calculate the mean log views per channel'''
        print("Calculating channel mean log views...")
        means_dict = df.groupby('channel')['log_views'].mean().to_dict()
        return means_dict
        
    
    def _add_channel_mean_log_views(self, df, test=False):
        '''
        Helper function to add channel mean log views. The means_dict can be provided if the df does not contain the views column but has the same channels.
        When test=True, it uses the means_dict to channel means from training data.
        '''
        if  test and 'channel_mean_log_views' not in df.columns:
            print("Adding channel mean log views to test by looking at train...")
            means_dict = self._get_channel_mean_log_views(pd.read_parquet('data/processed/enriched_train_val.parquet'))
            df['channel_mean_log_views'] = df['channel'].map(means_dict)
        elif 'channel_mean_log_views' not in df.columns:
            print("Adding channel mean log views...")
            channel_mean_log_views = df.groupby('channel')['log_views'].mean().to_dict()
            df['channel_mean_log_views'] = df['channel'].map(channel_mean_log_views)
        return df
    
    def _add_anomaly_class(self, df, test=False):
        '''Helper function to calculate the anomaly class, and add the one-hot encoded version of it.'''
        if test and 'anomaly_class' not in df.columns:
            print("Adding anomaly class as nan...")
            df['anomaly_class'] = np.nan
            df['encoded_anomaly_class'] = np.nan
            return df
        if 'anomaly_class' not in df.columns:
            print("Adding anomaly class...")
            df['anomaly_class'] = (df['log_views'] - df['channel_mean_log_views']).round()
            df['anomaly_class'] = df['anomaly_class'].astype(int)
            df['anomaly_class'] = df['anomaly_class'].clip(lower=-5, upper=6) 
        if 'encoded_anomaly_class' not in df.columns:
            print("Adding one-hot encoded anomaly class...")
            all_classes = list(range(-5, 7))
            one_hot = pd.get_dummies(df['anomaly_class'])
            one_hot = one_hot.reindex(columns=all_classes, fill_value=0)
            df['encoded_anomaly_class'] = one_hot.values.tolist()
        return df
            
    def _add_views_class(self, df, test=False):
        '''
        Helper function to add the views class, which is just the rounded log_views of the video.
        The point of the views_class was to be able to ablation studies, removing the residual conection. 
        '''
        if test and 'views_class' not in df.columns:
            df['views_class'] = np.nan  
            df['encoded_views_class'] = np.nan
            return df
        if 'views_class' not in df.columns: 
            print("Adding views_class")
            df['views_class'] =  df['log_views'].round()
            df['views_class'] = df['views_class'].astype(int)
            df['views_class'] = df['views_class'].clip(lower=5, upper=16)
        if 'encoded_views_class' not in df.columns:
            print('Adding one-hot encoded views class')
            all_classes = list(range(5, 17))
            one_hot = pd.get_dummies(df['views_class'])
            one_hot = one_hot.reindex(columns=all_classes, fill_value=0)
            df['encoded_views_class'] = one_hot.values.tolist()
        return df     
    
    def _add_guess_factor(self, df):
        '''
        Helper function to add the guess factor.
        This feature encodes if a channel was basically absent from the training set. 
        The idea was to use this feature in order to weaken the participation of "channel_mean_log_views" 
        in the prediction. Preliminary results were inconclusive, hence we ended up abandonning the feature in our models. 
        '''
        if 'guess_factor' not in df.columns:
            print("Adding guess factor...")
            channel_counts = df['channel'].value_counts()
            df['guess_factor'] = df['channel'].map(lambda x: channel_counts[x]/20 if channel_counts[x] < 20 else 1.0)
        return df
    
    def _classify_image_short(self, image_path, brightness_threshold_0_255=10, num_rows_to_check=10):
        '''
        Helper function to classify images as shorts.
        The classification logic is pretty simple : almost all non-short vides after 2015
        have black bars on the top and bottom of the thumbnail. This corresponds to a thumbnail
        resolution change around 2015. Yet shorts don't have those black bars (since they fit 
        the mobile format) and only appear in 2020. 
        '''
        try:
            gray_tensor = io.read_image(image_path, mode=io.ImageReadMode.GRAY)
        except Exception as e:
            print(f"Skipping image {image_path} due to load error: {e}")
            return None
        if gray_tensor.shape[0] > 1:
            gray_tensor = gray_tensor[0, :, :].unsqueeze(0)
        _, h, _ = gray_tensor.shape
        gray_np = gray_tensor.squeeze().numpy()
        top_rows = gray_np[0:num_rows_to_check, :]
        bottom_rows = gray_np[h-num_rows_to_check:h, :]
        mean_brightness_top = np.mean(top_rows)
        mean_brightness_bottom = np.mean(bottom_rows)
        if mean_brightness_top < brightness_threshold_0_255 and \
           mean_brightness_bottom < brightness_threshold_0_255:
            return 0
        else:
            return 1
        
    
    def _add_short(self, df):
        '''Helper function to add short label.'''
        if 'short' not in df.columns:
            print("Determining if videos are short...")
            df['short'] = df['id'].apply(lambda x: self._classify_image_short(os.path.join(self.train_val_image_path, f"{x}.jpg")))
        return df
            
    def _add_topic_encoding(self, df):
        '''
        Helper function to add topic encoding using SentenceTransformer. 
        The output is a 384-dimensional vector for each string : 'title ||| channel' .
        The topic encoding is then used to calculate flashiness
        '''
        if 'topic_embedding' not in df.columns:
            print("Adding topic embedding...")
            df['topic'] = df['title'] + ' ||| ' + df['channel']
            topic_embeddings = self.sentence_transformer.encode(df['topic'].tolist(), show_progress_bar=True, device=self.device).astype(np.float32)
            df['topic_embedding'] = list(topic_embeddings)
        return df
    
    def _add_flashiness(self, df, test=False):
        '''
        Helper function to calculate thumbnail flashiness.
        Flashiness is given by a video's views divided by the distance wieghted average of
        it's 100 nearest neighbor's views. The k-nn search was done using faiss. 
        '''
        if test:
            df['flashiness'] = np.zeros(len(df))
            df['log_flashiness'] = np.zeros(len(df))
            return df
        if 'flashiness' not in df.columns:
            topic_embeddings = np.array(df['topic_embedding'].tolist(), dtype=np.float32)
            self.index.add(topic_embeddings)
            distances_sq, indices = self.index.search(topic_embeddings, 100+1)
            flashiness_scores = []
            for i in tqdm(range(len(df))):
                current_video_views = df.iloc[i]['views']
                neighbors = indices[i, 1:]
                neighbor_distances = distances_sq[i, 1:]
                neighbor_views = df.iloc[neighbors]['views'].values
                sum_distances = neighbor_distances.sum()
                if  sum_distances > 0:
                    mean_neighbor_views = np.dot(neighbor_views, neighbor_distances) / sum_distances
                else:
                    mean_neighbor_views = neighbor_views.mean()  # fallback if all distances are zero
                flashiness_scores.append(current_video_views/mean_neighbor_views)
            df['flashiness'] = flashiness_scores
            df['log_flashiness'] = np.log(flashiness_scores)
            return df
            
            
    
    def enrich_train_val(self, from_zero=True):
        '''Main function to enrich the train/val dataset.'''
        if not from_zero:
            output_path = os.path.join(self.output_file, 'enriched_train_val.parquet')
            if os.path.exists(output_path):
                print("Enriching existing enriched train/val data.")
                df = pd.read_parquet(output_path)
            else:
                print("Enriching train/val data from scratch.")
                from_zero = True
        if from_zero:
            print("Enriching train/val data from scratch.")
            df = pd.read_csv(self.train_val_csv)
        
        df = self._add_date_features(df)
        df = self._add_channel_int(df)
        df = self._add_title_embedding(df)
        df = self._add_image_embedding(df, self.train_val_image_path)
        df = self._add_log_views(df)
        df = self._add_channel_mean_log_views(df)
        df = self._add_anomaly_class(df)
        df = self._add_guess_factor(df)
        df = self._add_short(df)
        df = self._add_topic_encoding(df)
        df = self._add_views_class(df)
        df = self._add_flashiness(df)
        
        # Save the enriched DataFrame
        output_path = os.path.join(self.output_file, 'enriched_train_val.parquet')
        df.to_parquet(output_path, index=False)
        print(f"Enriched train/val data saved to {output_path}")
    
    def enrich_test(self, from_zero=True):
        '''Main function to enrich the test dataset.'''
        if not from_zero:
            output_path = os.path.join(self.output_file, 'enriched_test.parquet')
            if os.path.exists(output_path):
                print("Enriching existing enriched test data.")
                df = pd.read_parquet(output_path)
            else:
                print("Enriching test data from scratch.")
                from_zero = True
        if from_zero:
            print("Enriching test data from scratch.")
            df = pd.read_csv(self.test_csv)
        
        df = self._add_date_features(df)
        df = self._add_channel_int(df, test=True)
        df = self._add_title_embedding(df)
        df = self._add_image_embedding(df, self.test_image_path)
        df = self._add_log_views(df, test=True)
        df = self._add_anomaly_class(df, test=True)
        df = self._add_channel_mean_log_views(df, test=True)
        df = self._add_guess_factor(df)
        df = self._add_short(df)
        df = self._add_views_class(df, test=True)
        df = self._add_flashiness(df, test=True)
        
        # Save the enriched DataFrame
        output_path = os.path.join(self.output_file, 'enriched_test.parquet')
        df.to_parquet(output_path, index=False)
        print(f"Enriched test data saved to {output_path}")
        
if __name__ == '__main__':
    enricher = Enricher()
    enricher.enrich_train_val(from_zero=False)
    enricher.enrich_test(from_zero=False)
    
    
    
# the classes in train and in test will need to have the same integer encoding, so we need to fit the label encoder on the train data and then use it on the test data
        