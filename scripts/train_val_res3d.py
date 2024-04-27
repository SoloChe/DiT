import sys
import os
sys.path.append(os.path.realpath('./'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam
from models.resnet_3d import generate_model
from utils import create_logger
from data_med import BrainDataset_3D, get_age, normalise_percentile
import argparse


def main(args):

    # Set up logging:
    os.makedirs(args.logging_dir, exist_ok=True)
    logger, _ = create_logger(args.logging_dir, ddp=False, tb=False)
    logger.info(args)
    

    # Set up device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # dataset and loss config
    prefix = None if args.prefix == 'all' else args.prefix
    crop = args.crop
    transform = normalise_percentile if args.use_trans else None
        
    if args.loss_type == 1: # classification
        _, age_map, _ = get_age(args.age_path, prefix=prefix, round_age=True) # age:index
        reversed_age_map = {v: k for k, v in age_map.items()} # index:age
        num_classes = len(age_map)
        logger.info(f"Number of classes: {num_classes}")
    
        loss_fn = nn.CrossEntropyLoss()
        model = generate_model(
            model_depth=args.model_depth, n_classes=num_classes, n_input_channels=1
        )
        train_dataset = BrainDataset_3D(args.data_path, args.age_path, mode="train", transform=transform, prefix=prefix, round_age=True, crop=crop)
        val_dataset = BrainDataset_3D(args.data_path, args.age_path, mode="val",transform=transform, prefix=prefix, round_age=True, crop=crop)
    elif args.loss_type == 2: # regression
        loss_fn = nn.MSELoss()
        model = generate_model(
            model_depth=args.model_depth, n_classes=1, n_input_channels=1
        )
        train_dataset = BrainDataset_3D(args.data_path, args.age_path, mode="train", transform=transform, prefix=prefix, round_age=False, crop=crop)
        val_dataset = BrainDataset_3D(args.data_path, args.age_path, mode="val", transform=transform, prefix=prefix, round_age=False, crop=crop)
    elif args.loss_type == 3: # regression
        loss_fn = nn.L1Loss()
        model = generate_model(
            model_depth=args.model_depth, n_classes=1, n_input_channels=1
        )
        train_dataset = BrainDataset_3D(args.data_path, args.age_path, mode="train", transform=transform, prefix=prefix, round_age=False, crop=crop)
        val_dataset = BrainDataset_3D(args.data_path, args.age_path, mode="val", transform=transform, prefix=prefix, round_age=False, crop=crop)
    else:
        raise ValueError("Invalid loss type.")
    
    model = model.to(device)
    # opt = AdamW(model.parameters(), lr=args.lr)
    opt = Adam(model.parameters(), args.lr)

    logger.info(
        f"Model with {sum(p.numel() for p in model.parameters()) / 1024 / 1024:.2f}M parameters."
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_path})")
    logger.info(f"Validation dataset contains {len(val_dataset):,} images ({args.data_path})")
    
    train_steps = 0
    running_loss = 0
    for epoch in range(args.epochs):
        # training loop
        for data in train_loader:
            train_steps += 1
            
            x = data[0].to(device)
            if args.loss_type == 1:
                y = data[1].to(device) # age_index
            else:
                y = data[3].float().to(device) # age
                
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            
            running_loss += loss.item()

            # evaluation loop
            if train_steps % args.val_freq == 0:
                logger.info(f"Epoch {epoch} Train Steps {train_steps}, running_train_loss={(running_loss/(args.val_freq)):.4f}")
                running_loss = 0
        
                model.eval()
                with torch.no_grad():
                    MAE = []
                    total_val_loss = 0
                    for data in val_loader:
                        x = data[0].to(device)
                        if args.loss_type == 1:
                            y = data[1].to(device) # age_index
                            logits = model(x)
                            val_loss = loss_fn(logits, y)
                            pred_age_index = torch.argmax(logits, dim=1)
                            # select from reversed_age_map
                            pred_ages = torch.tensor([reversed_age_map[i.item()] for i in pred_age_index], dtype=torch.float32).to(device)
                            true_ages = torch.tensor([reversed_age_map[i.item()] for i in y], dtype=torch.float32).to(device)
                            MAE.append(torch.abs(pred_ages - true_ages))
                        else:
                            y = data[3].float().to(device) # age
                            y = y.reshape(-1, 1)
                            pred_age = model(x)
                            val_loss = loss_fn(pred_age, y)
                            MAE.append(torch.abs(pred_age - y).ravel())
                        
                        total_val_loss += val_loss.item()

                    MAE = torch.cat(MAE)
                    MAE_mean = torch.mean(MAE)
                    logger.info(f"Epoch {epoch} Train Steps {train_steps}, ave_val_loss = {total_val_loss/len(val_loader):.4f}")
                    logger.info(f"Epoch {epoch} Train Steps {train_steps}, MAE_mean={MAE_mean:.4f}")
                    logger.info(f"Epoch {epoch} Train Steps {train_steps}, MAE {MAE}")
                model.train()
                
        if epoch % args.save_freq == 0:
            checkpoint = {
                "epoch": epoch,
                "train_steps": train_steps,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
            }
            torch.save(checkpoint, f"{args.logging_dir}/model_{epoch}.pth")
                
    logger.info("Finished training.")
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/data/amciilab/yiming/DATA/brain_age/extracted/")
    parser.add_argument("--age_path", type=str, default="/data/amciilab/yiming/DATA/brain_age/masterdata.csv")
    parser.add_argument("--logging_dir", type=str, default="logs_res/")
    parser.add_argument("--model_depth", type=int, default=18)
    parser.add_argument("--loss_type", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--prefix", type=str, default='all')
    parser.add_argument("--crop", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--use_trans", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--val_freq", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=10)
    args = parser.parse_args()
    
    main(args)
    
        
        

         
       
