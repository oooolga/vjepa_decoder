import torch
import numpy as np
import gradio as gr
import sys
from datetime import datetime

import torch, torch.distributed as dist
import torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.transforms import Compose, ToTensor, Resize
from torch.nn.parallel import DistributedDataParallel as DDP
from zmq import device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amp_dtype = torch.float16
scaler = torch.amp.GradScaler()    # CUDA & ROCm; no scaler needed for bf16

import wandb
epochs = 50
batch_size = 1
lr = 1e-4
log_freq = 100 # Frequency of logging to wandb

run = wandb.init(
    project="VJEPA-decoder-training",          # shows up as a tab in the UI
    name="exp‑001_bs{}_lr{}".format(batch_size, lr),         # optional, human‑readable
    config={                            # anything you want to remember
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "amp": True,
    },
)

sys.path.append("vjepa/")

from vjepa.models.model import VJEPAAutoEncoder
from vjepa.dataloader.davis_loader import DAVIS2017Dataset

def main():
    global optim
    multiprocessing = False # Set to True if using multiple GPUs
    if multiprocessing:
        dist.init_process_group("nccl")          # ① initialises collective ops
        rank = dist.get_rank()
        torch.cuda.set_device(rank)              # ② pin one GPU per process

    model = VJEPAAutoEncoder(image_size=(476, 854))
    if multiprocessing:
        model = model.cuda(rank)
        model.eval()
        # Wrap the model with DDP
        ddp_model = DDP(model, device_ids=[rank])
    else:
        model = model.cuda()

    # Define transform for both image and mask
    transform = Compose([
        Resize((476, 854)),  # Resize to match model input
        ToTensor()
    ])
    # Initialize the dataset
    davis_dataset = DAVIS2017Dataset(root_dir="/network/scratch/x/xuolga/Datasets/davis/DAVIS",
                                     split="train",
                                     transform=transform,
                                     overlap=24)
    if multiprocessing:
        sampler = DistributedSampler(davis_dataset, shuffle=True)

        dataloader = DataLoader(
            davis_dataset,
            batch_size=batch_size,          # Change as needed
            # shuffle=True,        # Optional
            num_workers=4,         # Adjust based on CPU cores
            pin_memory=True,       # Good for CUDA
            sampler=sampler
        )
    else:
        dataloader = DataLoader(
            davis_dataset,
            batch_size=batch_size,          # Change as needed
            shuffle=True,          # Optional
            num_workers=4,         # Adjust based on CPU cores
            pin_memory=True        # Good for CUDA
        )
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    if multiprocessing:
        optim = optim.Adam(ddp_model.parameters(), lr=lr)
    else:
        optim = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optim, step_size=10, gamma=0.1)

    for epoch in range(epochs):
        for i, batch in enumerate(dataloader):
            images, _ = batch  # each is shape [B, C, H, W]

            optim.zero_grad(set_to_none=True)      # avoids touching freed memory
            
            # === autocast =========================================
            # Use autocast for mixed precision training
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                # === forward =========================================
                decoded_result = ddp_model(images.cuda(rank, non_blocking=True)) if multiprocessing else model(images.cuda())
                # === loss ============================================
                # Calculate loss
                loss = criterion(decoded_result, images.cuda(rank, non_blocking=True)) if multiprocessing else criterion(decoded_result, images.cuda())
            # === backward ========================================
            scaler.scale(loss).backward()  # Scales the loss to avoid underflow
            scaler.step(optim)             # Updates the model parameters
            scaler.update()                # Updates the scale for the next iteration

            print(f"Epoch: {epoch+1}/{epochs}\tIteration: {i}/{len(dataloader)}\tLoss: {loss.item()}")
            if i % log_freq == 0:
                log_images = [
                    wandb.Image((img.clamp(min=0.0, max=1.0) * 255).round().to(torch.uint8)) for img in decoded_result[0].cpu()
                ]
                wandb.log({"iteration": i + epoch * len(dataloader),
                           "samples": log_images,
                           "loss": loss.item()})
        
        scheduler.step()  # Update learning rate
        print(f"Epoch {epoch+1}/{epochs} completed. Learning rate: {scheduler.get_last_lr()[0]}")
        wandb.log({"epoch": epoch + 1,
                   "learning_rate": scheduler.get_last_lr()[0]})  # Log a sample image

    # Save the model
    if multiprocessing:
        torch.save(ddp_model.state_dict(), "vjepa_ae.pth")
    else:
        torch.save(model.state_dict(), "vjepa_ae.pth")
    print("Training completed and model saved.")
    wandb.finish()  # Finish the wandb run

if __name__ == "__main__":
    main()
 