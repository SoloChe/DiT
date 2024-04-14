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
    logger, _ = create_logger(args.logging_dir, ddp=False, tb=False)
    logger.info(args)

    # Set up device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = generate_model(
        model_depth=args.model_depth, n_classes=args.num_classes, n_input_channels=1
    )
    model = model.to(device)

    opt = AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    logger.info(
        f"Model with {sum(p.numel() for p in model.parameters()) / 1024 / 1024:.2f}M parameters."
    )

    _, age_map, _ = get_age(args.age_path) # age:index
    reversed_age_map = {v: k for k, v in age_map.items()} # index:age
    
    train_dataset = BrainDataset_3D(args.data_path, args.age_path, mode="train")
    val_dataset = BrainDataset_3D(args.data_path, args.age_path, mode="val")
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
            y = data[1].to(device)

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
                    y = data[1].to(device)
                    logits = model(x)
                    pred_age_index = torch.argmax(logits, dim=1)
                    
                    # select from reversed_age_map
                    pred_ages = torch.tensor([reversed_age_map[i.item()] for i in pred_age_index], dtype=torch.float32)
                    pred_ages = pred_ages.to(device)
                    
                    mae = torch.abs(pred_ages - y.float())
                    MAE.append(mae)
                   
                MAE = torch.cat(MAE)
                print(f'MAE shape{MAE.shape}')
                MAE = torch.mean(MAE)
                      
                logger.info(f"Epoch {epoch}, mae={MAE:.4f}")
                
    logger.info("Finished training.")
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/data/amciilab/yiming/DATA/brain_age/extracted/")
    parser.add_argument("--age_path", type=str, default="/data/amciilab/yiming/DATA/brain_age/masterdata.csv")
    parser.add_argument("--logging_dir", type=str, default="logs_res/")
    parser.add_argument("--model_depth", type=int, default=50)
    parser.add_argument("--num_classes", type=int, default=65)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--val_freq", type=int, default=1)
    args = parser.parse_args()
    
    main(args)
    
        
        

         
       
