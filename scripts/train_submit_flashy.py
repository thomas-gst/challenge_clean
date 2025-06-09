from src.data.dataset import VideoDataset
from src.data.stratify import straitfy_validation_set
from src.models.flashy import  FlashyFinetune
from src.models.fusion import MultiTaskFusion
from src.utils.train import Trainer
from src.utils.loss import flashy_Huber, multi_task_loss

from torch.utils.data import DataLoader
import torch.nn as nn 
from torchvision import transforms
from transformers import BertConfig
import pandas as pd
import multiprocessing
from sklearn.model_selection import train_test_split
import wandb
import torch
import numpy as np
from transformers import get_linear_schedule_with_warmup


## Flashy encoder hyperparameters
# Training hyperparameters
FLASHY_BATCH_SIZE = 128
FLASHY_EPOCHS = 20
FLASHY_LEARNING_RATE = 1e-3
FLASHY_WEIGHT_DECAY = 1e-5
FLASHY_STRATIFY = False
FLASHY_VAL_SIZE = 0.07
UNFREEZE_EPOCH = 5


# Vision Model hyperparameters
FLASHY_EMBEDDING_DIM = 128
FLASHY_REGRESSOR = nn.Sequential(
    nn.BatchNorm1d(2048),
    nn.Linear(2048,FLASHY_EMBEDDING_DIM),
    nn.ReLU(),
    
    nn.BatchNorm1d(FLASHY_EMBEDDING_DIM),
    nn.Dropout(0.2),
    nn.Linear(FLASHY_EMBEDDING_DIM, 1)
)

## Fusion Model hyperparameters
# Training hyperparameters
BATCH_SIZE = 64
EPOCHS = 155
LEARNING_RATE =5e-5
WEIGHT_DECAY = 1e-2
STRATIFY = False
VAL_SIZE = 0.01
START_ALPHA = 0
TASK_PERIOD = list(range(0, 10))

# Transformer hyperparameters
EMBEDDING_DIM = 128
NUM_LAYERS = 4
NUM_HEADS = 8
INTERMEDIATE_FACTOR = 2
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
REGRESSOR_DROPOUT = 0.4
REGRESSOR = nn.Sequential(
    nn.BatchNorm1d(EMBEDDING_DIM+11),
    nn.Dropout(0.3),
    nn.Linear(EMBEDDING_DIM+11, EMBEDDING_DIM+11),
    nn.ReLU(),
    nn.BatchNorm1d(EMBEDDING_DIM+11), 
    nn.Dropout(0.1), 
    nn.Linear(EMBEDDING_DIM+11, 1)
)


def main():
    
    ## Flashy encoder training
    
    wandb.init(
        project="Youtube views challenge",
        name="resnet-fine_tune",
        config= {
            'batch_size': FLASHY_BATCH_SIZE,
            "epochs": FLASHY_EPOCHS,
            "learning_rate": FLASHY_LEARNING_RATE,
            "weight_decay" : FLASHY_WEIGHT_DECAY,
            "stratify": FLASHY_STRATIFY,
            "val_size": FLASHY_VAL_SIZE,
            "model": "flashy_encoder",
            "embedding_dim": FLASHY_EMBEDDING_DIM, 
        }
    )
    df_train_val = pd.read_parquet("data/processed/enriched_train_val.parquet")
    df_train_val = df_train_val[~np.isinf(df_train_val["log_flashiness"])]
    df_test = pd.read_parquet("data/processed/enriched_test.parquet")

    if FLASHY_STRATIFY:
        df_train, df_val = straitfy_validation_set(df_train_val, df_test, val_size=FLASHY_VAL_SIZE)
    else:
        df_train, df_val = train_test_split(df_train_val, test_size=FLASHY_VAL_SIZE)


    print(f"Train set size: {len(df_train)}")
    print(f"Validation set size: {len(df_val)}")
    print(f"Test set size: {len(df_test)}")
    
    model = FlashyFinetune(embedding_dim=FLASHY_EMBEDDING_DIM, regressor=FLASHY_REGRESSOR)
    
    
    transform = transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),                      
        transforms.Normalize(                       
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    train_dataset = VideoDataset(df_train, model.features, model.targets, image_path="data/raw/train_val/", transform=transform)
    val_dataset = VideoDataset(df_val, model.features, model.targets, image_path="data/raw/train_val/", transform=transform)
    test_dataset = VideoDataset(df_test, model.features, model.targets, image_path="data/raw/test/", test=True, transform=transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=FLASHY_BATCH_SIZE,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=FLASHY_BATCH_SIZE,
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
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=FLASHY_LEARNING_RATE, weight_decay=FLASHY_WEIGHT_DECAY
    )
    
    loss_fn = flashy_Huber()
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
        unfreeze_epoch = UNFREEZE_EPOCH,
    )
    trainer.train(epochs=FLASHY_EPOCHS, calculate_mse=False)
    model.save_model("models/flashy_encoder.pth")
    wandb.finish()

    
## Fusion training and submission
    wandb.init(
        project="YouTube views challenge",
        name="fusion-run-tuned",
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

    model = MultiTaskFusion(
        embedding_dim=EMBEDDING_DIM,
        token_dropout=TOKEN_DROPOUT,
        transformer_config=TRANSFORMER_CONFIG,
        classifier=CLASSIFIER,
        classifier_dropout=CLASSIFIER_DROPOUT,
        regressor=REGRESSOR,
        regressor_droupout=REGRESSOR_DROPOUT,
        image_embedding_dim=2048
    )

    train_val_encoding_df, test_encoding_df = trainer.calculate_encodings()
    train_val_encoding_df['id'] = train_val_encoding_df['id'].astype(str)
    test_encoding_df['id'] = test_encoding_df['id'].astype(int)

    
    df_train_val = df_train_val.drop(columns=["image_embedding"], errors="ignore")
    df_test = df_test.drop(columns=["image_embedding"], errors="ignore")
    df_train_val['id'] = df_train_val['id'].astype(str)
    df_test['id'] = df_test['id'].astype(int)
    

    df_train_val = df_train_val.merge(train_val_encoding_df, on="id", how="left")
    df_test = df_test.merge(test_encoding_df, on="id", how="left")
    df_train_val = df_train_val.dropna(subset=["image_embedding"])
    df_test = df_test.dropna(subset=["image_embedding"])
    print(df_test[['id','image_embedding']].head())
    print(df_train_val[['id','image_embedding']].head())
    
    if STRATIFY:
        df_train, df_val = straitfy_validation_set(df_train_val, df_test, val_size=VAL_SIZE)
    else:
        df_train, df_val = train_test_split(df_train_val, test_size=VAL_SIZE)

    del df_train_val  
    
    print(f"Train set size: {len(df_train)}")
    print(f"Validation set size: {len(df_val)}")
    print(f"Test set size: {len(df_test)}")
    
    train_dataset = VideoDataset(df_train, model.features, model.targets, image_path="data/raw/train_val/", transform=transform)
    val_dataset = VideoDataset(df_val, model.features, model.targets, image_path="data/raw/train_val/", transform=transform)
    test_dataset = VideoDataset(df_test, model.features, model.targets, image_path="data/raw/test/", test=True, transform=transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4,
        persistent_workers=True,
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
    trainer.calculate_submission("submissions/flashy_fusion_submission.csv")
    model.save_model("models/flashy_fusion.pth")
    wandb.finish()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
