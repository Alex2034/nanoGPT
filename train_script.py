import argparse
import time
import datetime
import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from model_hyperbolic import GPT, GPTConfig  

# Argument parsing
parser = argparse.ArgumentParser(description="Train nanoGPT on Tao Te Ching")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (default: 32)")
parser.add_argument("--block_size", type=int, default=32, help="Context size (default: 32)")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train (default: 10)")
parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate (default: 0.003)")
parser.add_argument("--n_embd", type=int, default=16, help="Embedding dimension (default: 16)")
parser.add_argument("--mode", type=str, default='original', help="Attention mode (default: original)")
args = parser.parse_args()

# Dataset preparation
class TaoTeChingDataset(Dataset):
    def __init__(self, data, block_size):
        chars = sorted(list(set(data)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_size = block_size
        self.data = [self.stoi[ch] for ch in data]
        self.vocab_size = len(chars)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        dix = torch.tensor(chunk[:-1], dtype=torch.long)
        target = torch.tensor(chunk[1:], dtype=torch.long)
        return dix, target

# FULL TAO DATASET

# with open('data/tao.txt', 'r', encoding='utf-8') as f:
#     full_data = f.read()
# chars = sorted(list(set(full_data)))
# full_dataset = TaoTeChingDataset(full_data, chars, block_size=args['block_size'])

# split_idx = int(len(full_data) * 0.9)

# train_dataset = TaoTeChingDataset(full_data[:split_idx], chars, block_size=args['block_size'])
# val_dataset = TaoTeChingDataset(full_data[split_idx:], chars, block_size=args['block_size'])

# train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args['batch_size'])
# val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args['batch_size'])  


# SMALL TAO DATASET

with open('data/tao.txt', 'r', encoding='utf-8') as f:
    data = f.read()
    chars = sorted(list(set(data)))
    overfit_data = data[17:301]
    overfit_val_data = data[306:474]
    
overfit_dataset = TaoTeChingDataset(overfit_data, block_size=args.block_size)
overfit_loader = DataLoader(overfit_dataset, batch_size=args.batch_size, shuffle=True)
    
overfit_val_dataset = TaoTeChingDataset(overfit_val_data, block_size=args.block_size)
overfit_val_loader = DataLoader(overfit_val_dataset, batch_size=args.batch_size, shuffle=False)

# Device setup
if torch.cuda.is_available():
    gpu_id = '0' # select a single GPU
    # gpu_id = '2,3' # select multiple GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda")
    print('GPU name: {:s}, gpu_id: {:s}'.format(torch.cuda.get_device_name(0),gpu_id))

else:
    device = torch.device("cpu")
    gpu_id = -1 # select CPU
    
    
# Model setup

model_args = dict(n_layer=6, 
                  n_head=8, 
                  n_embd=args.n_embd, 
                  block_size=args.block_size, 
                  bias=False, 
                  vocab_size=overfit_dataset.vocab_size, 
                  dropout=0.0, 
                  mode=args.mode)

model = GPT(GPTConfig(**model_args)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

# New Training loop
def train(model, epoch, loader):
    model.train()
    train_loss = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        _, loss = model(data, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
#         if idx % 100 == 0:
#             print(f"Epoch: {epoch} | Loss: {loss.item()}")
    train_loss = train_loss/len(loader)
    return train_loss
        
            
def evaluate(model, epoch, loader):
    model.eval()  
    val_loss = 0
    with torch.no_grad():  
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            _, loss = model(data, target)
            val_loss += loss.item()  
            
    val_loss /= len(loader) 
    return val_loss

def save_checkpoint(args, model, optimizer, model_args, train_losses, val_losses, filename, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'train_losses': train_losses,
        'val_losses': val_losses,
    }
    torch.save(checkpoint, os.path.join(out_dir, filename))
#     print(f"Checkpoint saved to {filename}")

train_losses = []
val_losses = []
save_checkpoints = True
out_dir = 'out'

# filename for saving
date_time_str = datetime.datetime.now().strftime("%m.%d, %H-%M")
dim = model_args['n_embd'] // model_args['n_head']
mode = args.mode
filename = f"{date_time_str}, {dim}-dim, {mode}, {args.epochs} epochs ckpt.pt"

for epoch in range(1, args.epochs + 1):
    start_time = time.time()
    train_loss = train(model, epoch, overfit_loader)
    train_losses.append(train_loss)

    val_loss = evaluate(model, epoch, overfit_val_loader)
    val_losses.append(val_loss)
#     print(f"Epoch: {epoch} | Training Loss: {train_loss} | Validation Loss: {val_loss}")

    end_time = time.time()
    if epoch % 100 == 0:
        print(f"Epoch {epoch} completed in {end_time - start_time:.2f} seconds")
    
        if save_checkpoints:
            save_checkpoint(args, model, optimizer, model_args, train_losses, val_losses, filename, out_dir)




# Old Training loop
# def train(epoch):
#     model.train()
#     for idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         logits, loss = model(data, target)
#         loss.backward()
#         optimizer.step()
        
#         if idx % 100 == 0:
#             print(f"Epoch: {epoch} | Loss: {loss.item()}")
            
# def evaluate(model, device, validation_loader):
#     model.eval()  # Set the model to evaluation mode
#     val_loss = 0
#     with torch.no_grad():  # No gradients needed for validation, saves memory and computations
#         for data, target in validation_loader:
#             data, target = data.to(device), target.to(device)
#             logits, loss = model(data, target)
#             val_loss += loss.item()  # Sum up batch loss
#     val_loss /= len(validation_loader.dataset)  # Average loss
#     return val_loss


# for epoch in range(1, args.epochs + 1):
#     train(epoch) 

#     val_loss = evaluate(model, device, validation_loader)  # Evaluate on the validation set
#     print(f"Epoch: {epoch}, Validation Loss: {val_loss}")

#     # Conditionally save checkpoints
#     if save_checkpoints:
#         date_time_str = datetime.datetime.now().strftime("%H-%M")
#         filename = f"{date_time_str}, {n_embd//n_heads}-dim, {mode} ckpt.pt"
#         checkpoint = {
#             'model': model.state_dict(),
#             'optimizer': optimizer.state_dict(),
#             'model_args': model_args,
#             'train_losses': train_losses,
#             'val_losses': [val_loss],  # Since it's a single value per epoch
#         }
#         os.makedirs(out_dir, exist_ok=True)
#         torch.save(checkpoint, os.path.join(out_dir, filename))
