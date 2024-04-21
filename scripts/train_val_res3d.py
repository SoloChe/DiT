import sys
import os
sys.path.append(os.path.realpath('./'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from models.resnet_3d import generate_model
from utils import create_logger
from data_med import BrainDataset_3D, get_age
import argparse


def main(args):

    # Set up logging:
    os.makedirs(args.logging_dir, exist_ok=True)
    logger, _ = create_logger(args.logging_dir, ddp=False, tb=False)
    logger.info(args)
    

    # Set up device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    _, age_map, _ = get_age(args.age_path) # age:index
    reversed_age_map = {v: k for k, v in age_map.items()} # index:age
    num_classes = len(age_map)

    if args.loss_type == 1: # classification
        loss_fn = nn.CrossEntropyLoss()
        model = generate_model(
            model_depth=args.model_depth, n_classes=num_classes, n_input_channels=1
        )
    elif args.loss_type == 2: # regression
        loss_fn = nn.MSELoss()
        model = generate_model(
            model_depth=args.model_depth, n_classes=1, n_input_channels=1
        )
    elif args.loss_type == 3: # regression
        loss_fn = nn.L1Loss()
        model = generate_model(
            model_depth=args.model_depth, n_classes=1, n_input_channels=1
        )
    else:
        raise ValueError("Invalid loss type.")
    
    model = model.to(device)
    opt = AdamW(model.parameters(), lr=1e-4)
   

    logger.info(
        f"Model with {sum(p.numel() for p in model.parameters()) / 1024 / 1024:.2f}M parameters."
    )

    
    
    train_dataset = BrainDataset_3D(args.data_path, args.age_path, mode="train", prefix=None)
    val_dataset = BrainDataset_3D(args.data_path, args.age_path, mode="val", prefix=None)
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

    for epoch in range(args.epochs):
        running_loss = 0
        for i, data in enumerate(train_loader):
            x = data[0].to(device)
            if args.loss_type == 1:
                y = data[1].to(device)
            else:
                y = data[1].float()
                y = y.to(device)

            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            
            running_loss += loss.item()

        logger.info(f"Epoch {epoch}, train_loss={(running_loss/(i+1)):.4f}")
        
        if epoch % args.val_freq == 0:
            with torch.no_grad():
                MAE = []
                for i, data in enumerate(val_loader):
                    x = data[0].to(device)
                    
                    if args.loss_type == 1:
                        y = data[1].to(device) # age_index
                        logits = model(x)
                        pred_age_index = torch.argmax(logits, dim=1)
                        
                        # select from reversed_age_map
                        pred_ages = torch.tensor([reversed_age_map[i.item()] for i in pred_age_index], dtype=torch.float32)
                        true_ages = torch.tensor([reversed_age_map[i.item()] for i in y], dtype=torch.float32)
                        pred_ages = pred_ages.to(device)
                        true_ages = true_ages.to(device)
                        mae = torch.abs(pred_ages - true_ages)
                        MAE.append(mae)
                    else:
                        y = data[3].to(device) # age
                        y = y.reshape(-1, 1)
                        pred_age = model(x)
                        MAE.append(torch.abs(pred_age - y))
               
                MAE = torch.cat(MAE)
                # print(f'mae shape{mae.shape}')
                # print(f'MAE shape{MAE.shape}')
                MAE = torch.mean(MAE)
                      
                logger.info(f"Epoch {epoch}, mae={MAE:.4f}")
        
        if epoch % args.save_freq == 0:
            checkpoint = {
                "epoch": epoch,
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
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--val_freq", type=int, default=1)
    parser.add_argument("--save_freq", type=int, default=20)
    args = parser.parse_args()
    
    main(args)
    
        
        

         
       
