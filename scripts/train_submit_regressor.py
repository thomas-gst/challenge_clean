from src.data.dataset import VideoDataset
from src.data.stratify import straitfy_validation_set
from src.models.regressor import StandardRegressor
from src.utils.train import Trainer
from src.utils.loss import regressor_MSE

from torch.utils.data import DataLoader
import pandas as pd
import multiprocessing
from sklearn.model_selection import train_test_split
import wandb
import torch

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-2
STRATIFY = True
VAL_SIZE = 0.2
EMBEDDING_DIM = 6


def main():
    
    wandb.init(
        project="YouTube views challenge",     
        name="regressor-run-stratified",  
        config={                     
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "val_size": VAL_SIZE,
            "model": "standard_regressor",
        }
    )
    
    df_train_val = pd.read_parquet("data/processed/enriched_train_val.parquet")
    df_test = pd.read_parquet("data/processed/enriched_test.parquet")
    
    if STRATIFY:
        df_train, df_val = straitfy_validation_set(df_train_val, df_test, val_size=VAL_SIZE)
    else:
        df_train, df_val = train_test_split(df_train_val, test_size=VAL_SIZE)
        
    
    del df_train_val #unload df_train_val from memory
    
    print(f"Train set size: {len(df_train)}")
    print(f"Validation set size: {len(df_val)}")
    print(f"Test set size: {len(df_test)}")
    
    model = StandardRegressor(embedding_dim=EMBEDDING_DIM)
    
    train_dataset = VideoDataset(df_train, model.features, model.targets, image_path="data/raw/train_val/")
    val_dataset = VideoDataset(df_val, model.features, model.targets, image_path="data/raw/train_val/")
    test_dataset = VideoDataset(df_test, model.features, model.targets, image_path="data/raw/test/")
    
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
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
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
        silent=False
    )
    
    trainer.train(epochs=EPOCHS, calculate_mse=False)
    trainer.calculate_submission("submissions/regressor_submission.csv")
    model.save_model("models/regressor.pth")
    wandb.finish()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()