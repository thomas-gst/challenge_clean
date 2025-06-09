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
import wandb
import torch
import numpy as np
from transformers import get_linear_schedule_with_warmup

# Training hyperparameters
BATCH_SIZE = 64
EPOCHS = 120
LEARNING_RATE =7e-3
WEIGHT_DECAY = 7e-2
STRATIFY = False
VAL_SIZE = 0.01
START_ALPHA = 0
TASK_PERIOD = [k for period in range(4) for k in range(period*20, period*20+1)]

# Transformer hyperparameters
EMBEDDING_DIM = 64
NUM_LAYERS = 1
NUM_HEADS = 4
INTERMEDIATE_FACTOR = 1
HIDDEN_DROPOUT = 0.4
TOKEN_DROPOUT = 0.35
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
REGRESSOR_DROPOUT = 0.4
REGRESSOR = nn.Sequential(
    nn.BatchNorm1d(EMBEDDING_DIM+11),
    nn.Dropout(0.5),
    nn.Linear(EMBEDDING_DIM+11, EMBEDDING_DIM+11),
    nn.ReLU(),
    nn.BatchNorm1d(EMBEDDING_DIM+11), 
    nn.Dropout(0.3), 
    nn.Linear(EMBEDDING_DIM+11, 1)
)



def main():
    wandb.init(
        project="YouTube views challenge",
        name="fusion-run-stratified",
        config={
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay" : WEIGHT_DECAY,
            "stratify": STRATIFY,
            "val_size": VAL_SIZE,
            "model": "multi_task_fusion",
            'start_alpha': START_ALPHA,
            'task_period' : TASK_PERIOD,
            'embedding_dim' : EMBEDDING_DIM,
            'num_layers' : NUM_LAYERS,
            'num_heads': NUM_HEADS,
            'intermediate_factor' : INTERMEDIATE_FACTOR,
            'hidden_dropout' : HIDDEN_DROPOUT,
            'token_dropout' : TOKEN_DROPOUT,
            'classifier_dropout': CLASSIFIER_DROPOUT,
            'regressor_dropout':REGRESSOR_DROPOUT,
        }
    )

    df_train_val = pd.read_parquet("data/processed/enriched_train_val.parquet")
    df_test = pd.read_parquet("data/processed/enriched_test.parquet")

    if STRATIFY:
        df_train, df_val = straitfy_validation_set(df_train_val, df_test, val_size=VAL_SIZE)
    else:
        df_train, df_val = train_test_split(df_train_val, test_size=VAL_SIZE)

    del df_train_val  # unload df_train_val from memory

    print(f"Train set size: {len(df_train)}")
    print(f"Validation set size: {len(df_val)}")
    print(f"Test set size: {len(df_test)}")

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
    loss_update_fn = loss_fn.update_alpha 
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
    trainer.calculate_submission("submissions/fusion_submission.csv")
    model.save_model("models/fusion.pth")
    wandb.finish()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()