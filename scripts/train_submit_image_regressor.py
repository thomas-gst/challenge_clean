from src.data.dataset import VideoDataset
from src.data.stratify import straitfy_validation_set
from src.models.image_regressor import  ImageRegressor
from src.utils.train import Trainer
from src.utils.loss import regressor_MSE

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
BATCH_SIZE = 128
EPOCHS = 150
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
STRATIFY = False
VAL_SIZE = 0.01


# Model hyperparameters
EMBEDDING_DIM = 32
RESIDUAL = False
REGRESSOR = nn.Sequential(
    nn.BatchNorm1d(768+11),
    nn.Linear(768+11,EMBEDDING_DIM),
    nn.ReLU(),
    
    nn.BatchNorm1d(EMBEDDING_DIM),
    nn.Dropout(0.2),
    nn.Linear(EMBEDDING_DIM, 1)
)

def main():
    wandb.init(
        project="Youtube views challenge",
        name="image-regression",
        config= {
            'batch_size': BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay" : WEIGHT_DECAY,
            "stratify": STRATIFY,
            "val_size": VAL_SIZE,
            "model": "image_regressor",
            "embedding_dim": EMBEDDING_DIM, 
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
    
    model = ImageRegressor(embedding_dim=EMBEDDING_DIM, residual=RESIDUAL, regressor=REGRESSOR)
    
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
    loss_fn = regressor_MSE()
    loss_update_fn = None
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
    trainer.calculate_submission("submissions/image_regression_submission.csv")
    model.save_model("models/fusion.pth")
    wandb.finish()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
    
        