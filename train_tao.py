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
# parser.add_argument("--n_embd", type=int, default=16, help="Embedding dimension (default: 16)")
parser.add_argument("--mode", type=str, default='original', help="Attention mode (default: original)")
parser.add_argument("--gpu", type=str, default='0')
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
    small_data = data[14:10095]
    small_val_data = data[10095:12900]
    
small_dataset = TaoTeChingDataset(small_data, block_size=args.block_size)
small_loader = DataLoader(small_dataset, batch_size=args.batch_size, shuffle=True)
    
small_val_dataset = TaoTeChingDataset(small_val_data, block_size=args.block_size)
small_val_loader = DataLoader(small_val_dataset, batch_size=args.batch_size, shuffle=False)

# Device setup
if torch.cuda.is_available():
    gpu_id = args.gpu # select a single GPU
    # gpu_id = '2,3' # select multiple GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda")
    print('GPU name: {:s}, gpu_id: {:s}'.format(torch.cuda.get_device_name(0),gpu_id))

else:
    device = torch.device("cpu")
    gpu_id = -1 # select CPU
    
    
# Model setup

n_layer = 8 
n_head = 8
n_embd = 24

model_args = dict(n_layer=n_layer, 
                  n_head=n_head, 
                  n_embd=n_embd, 
                  block_size=args.block_size, 
                  bias=False, 
                  vocab_size=small_dataset.vocab_size, 
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
    train_loss = train(model, epoch, small_loader)
    train_losses.append(train_loss)
#     print(f"Epoch: {epoch} | Training Loss: {train_loss} | Validation Loss: {val_loss}")

    end_time = time.time()
    
    
    if epoch % 100 == 0:
        val_loss = evaluate(model, epoch, small_val_loader)
        val_losses.append(val_loss)
        
    if epoch % 10 == 0:
        print(f"Epoch {epoch} completed in {end_time - start_time:.2f} seconds")
    
        if save_checkpoints:
            save_checkpoint(args, model, optimizer, model_args, train_losses, val_losses, filename, out_dir)

