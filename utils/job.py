from src.data.dataset import VideoDataset
from src.data.stratify import straitfy_validation_set
from src.models.fusion import MultiTaskFusion 
from src.utils.train import Trainer
from src.utils.loss import multi_task_loss

from torch.utils.data import DataLoader
import torch.nn as nn 
from transformers import BertConfig
import pandas as pd
import multiprocessing
from sklearn.model_selection import train_test_split
import torch
import numpy as np


'''
Script called by cross_validation.py
'''

import argparse 
def main(args):
    
    # Training hyperparameters
    BATCH_SIZE = 64
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    WEIGHT_DECAY = args.weight_decay
    VAL_SIZE = 0.01
    START_ALPHA = 0
    TASK_PERIOD = args.task_period

    # Transformer hyperparameters
    EMBEDDING_DIM = 128
    NUM_LAYERS = args.layers
    NUM_HEADS = args.heads
    INTERMEDIATE_FACTOR = args.intermediate_factor
    HIDDEN_DROPOUT = 0.35
    TOKEN_DROPOUT = 0.3
    TRANSFORMER_CONFIG = BertConfig(
        hidden_size=EMBEDDING_DIM,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        intermediate_size=EMBEDDING_DIM * INTERMEDIATE_FACTOR,
        hidden_dropout_prob=HIDDEN_DROPOUT,
        type_vocab_size=5, # cls + image + text + channel + short
        max_position_embeddings=5,
    )

    # Classifier hyperparameters
    CLASSIFIER_DROPOUT = 0.3
    CLASSIFIER = nn.Sequential(
        nn.BatchNorm1d(EMBEDDING_DIM),
        nn.Dropout(CLASSIFIER_DROPOUT+0.2),
        nn.Linear(EMBEDDING_DIM, 16),
        nn.ReLU(),
        
        nn.BatchNorm1d(16),
        nn.Dropout(CLASSIFIER_DROPOUT+0.2),
        nn.Linear(16, 16),
        nn.ReLU(),
        
        nn.BatchNorm1d(16),
        nn.Dropout(CLASSIFIER_DROPOUT+0.1),
        nn.Linear(16, 16),
        nn.ReLU(),
        
        nn.BatchNorm1d(16),
        nn.Dropout(CLASSIFIER_DROPOUT-0.1),
        nn.Linear(16, 12),   
    )
    
    # Regressor hyperparameters
    REGRESSOR_DROPOUT = 0.5
    REGRESSOR = nn.Sequential(
        nn.BatchNorm1d(EMBEDDING_DIM+11),
        nn.Dropout(0.5),
        nn.Linear(EMBEDDING_DIM+11, EMBEDDING_DIM+11),
        nn.ReLU(),
        nn.BatchNorm1d(EMBEDDING_DIM+11), 
        nn.Dropout(0.1), 
        nn.Linear(EMBEDDING_DIM+11, 1)
    )
    
    df_train_val = pd.read_parquet("data/processed/enriched_train_val.parquet")
    df_test = pd.read_parquet("data/processed/enriched_test.parquet")
    df_train, df_val = train_test_split(df_train_val, test_size=VAL_SIZE)
    
    model = MultiTaskFusion(
        embedding_dim=EMBEDDING_DIM,
        token_dropout=TOKEN_DROPOUT,
        transformer_config=TRANSFORMER_CONFIG,
        classifier=CLASSIFIER,
        classifier_dropout=CLASSIFIER_DROPOUT,
        regressor=REGRESSOR,
        regressor_droupout=REGRESSOR_DROPOUT,
    )
    
    train_dataset = VideoDataset(df_train, model.features, model.targets, image_path="data/raw/train_val/")
    val_dataset = VideoDataset(df_val, model.features, model.targets, image_path="data/raw/train_val/")
    test_dataset = VideoDataset(df_test, model.features, model.targets, image_path="data/raw/test/", test=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=4,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=4,
        persistent_workers=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = multi_task_loss(START_ALPHA, task_perdiod=TASK_PERIOD)
    loss_update_fn = loss_fn.update_alpha_periodic 
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        loss_update_fn=loss_update_fn,
        device=device,
        silent=False,
    )
    trainer.train(epochs=EPOCHS, calculate_mse=True)
    
    best_val_loss = min(trainer.val_loss_history)
    best_val_epoch = trainer.val_loss_indices[trainer.val_loss_history.index(best_val_loss)]  
    print(f"BEST_VAL_EPOCH:{best_val_epoch}")
    print(f"BEST_VAL_LOSS:{best_val_loss}")
    
    return best_val_loss




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the main script with specified arguments.")
    parser.add_argument('--epochs', type=int, default=80, help='Number of training epochs')
    parser.add_argument('--task_period', type=int, default=10, help='Number of training epochs for per task')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for the optimizer')
    parser.add_argument('--layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--heads', type=int, default=6, help='Number of attention heads in the transformer')
    parser.add_argument('--intermediate_factor', type=int, default=4, help='Intermediate factor for the transformer')
    
    args = parser.parse_args()
    
    main(args)
    
    