import torch 
import numpy as np
from sklearn.metrics import mean_squared_error
import wandb
from tqdm import tqdm
import pandas as pd

class Trainer():
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader, device, test_loader=None, silent=False, loss_update_fn=None, scheduler=None, unfreeze_epoch = -1):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        self.train_loss_history = [] 
        self.val_loss_history = []   
        
        self.silent = silent 
        self.loss_update_fn = loss_update_fn
        self.scheduler = scheduler
        self.unfreeze_epoch = unfreeze_epoch
        
        
        
    def _load_batch_to_device(self, data_dict):
        '''Helper function to load a batch of data to the specified device. For information, the data is a single dictionary per batch with keys as feature names and values as tensors.'''
        x = {k: data_dict[k].to(self.device) for k in self.model.features}
        target = {k: data_dict[k].to(self.device) for k in self.model.targets}
        return x, target
        
    
    def train_one_epoch(self):
        self.model.train()
        total_train_loss = 0
        num_samples_processed = 0
        for batch_dict in tqdm(self.train_loader, desc="Training", disable=self.silent):
            x, target = self._load_batch_to_device(batch_dict)
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            batch_size = target[self.model.targets[0]].size(0) # batch size calculation robust to drop_last = True
            total_train_loss += loss.item() * batch_size
            num_samples_processed += batch_size  
        avg_train_loss = total_train_loss / num_samples_processed
        self.train_loss_history.append(avg_train_loss)
        return avg_train_loss

    def validate_one_epoch(self):
        self.model.eval()
        total_val_loss = 0.0
        num_samples_processed = 0
        with torch.no_grad():
            for batch_dict in tqdm(self.val_loader, desc="Validating", disable=self.silent):
                x, target = self._load_batch_to_device(batch_dict)
                output = self.model(x)
                loss = self.loss_fn(output, target)
                batch_size = target[self.model.targets[0]].size(0) # batch size calculation robust to drop_last = True
                total_val_loss += loss.item() * batch_size
                num_samples_processed += batch_size 
        avg_val_loss = total_val_loss / num_samples_processed
        self.val_loss_history.append(avg_val_loss)
        return avg_val_loss

    def train(self, epochs, calculate_mse=False):
        self.model.to(self.device)
        for epoch in range(epochs):
            avg_train_loss = self.train_one_epoch()
            avg_val_loss = self.validate_one_epoch()
            if calculate_mse:
                val_mse = self.validation_mse()   
            if self.unfreeze_epoch >= 0 and self.unfreeze_epoch == epoch:
                self.model.unfreeze()        
            if self.loss_update_fn:
                self.loss_update_fn(epoch, avg_train_loss, avg_val_loss)
                print(f'Loss alpha updated to {self.loss_fn.alpha:.4f}')
            if not self.silent:
                print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, MSE: {val_mse:.4f}' if calculate_mse else f'Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss_epoch': avg_train_loss,
                    'val_loss_epoch': avg_val_loss,
                    'val_mse': val_mse if calculate_mse else None
                }) 
        if not self.silent:
            print(f'Training complete.')
            
    def validation_mse(self):
        self.model.eval()
        total_se = 0.0
        num_samples_processed = 0
        with torch.no_grad():
            for batch_dict in tqdm(self.val_loader, desc="Calculating Validation MSE", disable=self.silent):
                x, target = self._load_batch_to_device(batch_dict)
                prediction = self.model.predict(x)
                batch_mse = mean_squared_error(target['log_views'].cpu().numpy(), prediction.cpu().numpy())
                batch_size = target['log_views'].size(0) # batch size calculation robust to drop_last = True
                total_se += batch_mse * batch_size
                num_samples_processed += batch_size
        val_mse = total_se / num_samples_processed 
        return val_mse
        
        
            
    def calculate_submission(self, save_path):
        self.model.eval()
        predictions = []
        ids = []
        with torch.no_grad():
            for batch_dict in tqdm(self.test_loader, desc="Calculating Submission", disable=self.silent):
                x, _ = self._load_batch_to_device(batch_dict)
                prediction = self.model.predict(x).cpu().numpy().tolist()
                batch_ids = batch_dict['id'].numpy().astype(int).tolist()
                predictions.extend(prediction)
                ids.extend(batch_ids)
        
        predictions = np.clip(predictions, 0, None)
        predictions = np.exp(predictions)
        submission_df = pd.DataFrame({
            'ID': ids,
            'views': predictions
        })
        submission_df = submission_df.sort_values('ID')
        submission_df.to_csv(save_path, index=False)
        print('Submission file saved: '+save_path)
        return
    
    def calculate_encodings(self):
        train_encodings = []
        val_encodings = []
        test_encodings = []
        
        train_ids = []
        val_ids = []
        test_ids = []
        
        self.model.eval()
        
        with torch.no_grad():
            for batch_dict in tqdm(self.test_loader, desc="Calculating test encodings", disable=self.silent):
                x,_ = self._load_batch_to_device(batch_dict)
                encoding = self.model.encode(x).cpu().tolist()
            
                batch_ids = batch_dict['id']
                test_encodings.extend(encoding)
                test_ids.extend(batch_ids)
        test_encoding_df = pd.DataFrame({
            'id': test_ids, 
            'image_embedding': test_encodings,
        })   
        
        with torch.no_grad():
            for batch_dict in tqdm(self.train_loader, desc="Calculating train encodings", disable=self.silent):
                x,_ = self._load_batch_to_device(batch_dict)
                encoding = self.model.encode(x).cpu().tolist()
                batch_ids = batch_dict['id']
                train_encodings.extend(encoding)
                train_ids.extend(batch_ids)
        train_encoding_df = pd.DataFrame({
            'id':train_ids, 
            'image_embedding': train_encodings,
        })
        
        with torch.no_grad():
            for batch_dict in tqdm(self.val_loader, desc="Calculating val encodings", disable=self.silent):
                x,_ = self._load_batch_to_device(batch_dict)
                encoding = self.model.encode(x).cpu().tolist()
                batch_ids = batch_dict['id']
                val_encodings.extend(encoding)
                val_ids.extend(batch_ids)
        val_encoding_df = pd.DataFrame({
            'id': val_ids, 
            'image_embedding': val_encodings,
        })
            
        
        
        train_val_encoding_df = pd.concat([train_encoding_df, val_encoding_df], ignore_index=True)
        return train_val_encoding_df, test_encoding_df