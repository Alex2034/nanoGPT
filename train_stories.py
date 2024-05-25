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
parser = argparse.ArgumentParser(description="Train nanoGPT on TinyStories")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (default: 32)")
parser.add_argument("--block_size", type=int, default=32, help="Context size (default: 32)")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train (default: 10)")
parser.add_argument("--log_interval", type=int, default=100, help="Number of batches to log after (default: 100)")
parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate (default: 0.003)")
# parser.add_argument("--n_embd", type=int, default=16, help="Embedding dimension (default: 16)")
parser.add_argument("--mode", type=str, default='original', help="Attention mode (default: original)")
parser.add_argument("--gpu", type=str, default='0')
args = parser.parse_args()

# Assuming the structure and processing requirements are the same as TaoTeChingDataset
class TinyStoriesDataset(Dataset):
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

with open('data/TinyStories20mb.txt', 'r', encoding='utf-8') as f:
    data = f.read()
    # Assume similar split for training and validation, adjust indices as needed
    train_data = data[:int(len(data) * 0.15)]
    val_data = data[int(len(data) * 0.15): int(len(data) * 0.18)]

    # Create datasets
dataset = TinyStoriesDataset(train_data, block_size=args.block_size)
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

val_dataset = TinyStoriesDataset(val_data, block_size=args.block_size)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

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
                  vocab_size=dataset.vocab_size, 
                  dropout=0.0, 
                  mode=args.mode)

model = GPT(GPTConfig(**model_args)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8) 

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}min"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

# New Training loop
def train(model, epoch, loader, train_losses, log_interval=100):
    model.train()
    total_loss = 0.0
    total_batches = 0
    times = []
    start_time = time.time()

    for idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        _, loss = model(data, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        if (idx + 1) % log_interval == 0:
            average_loss = total_loss / total_batches
            train_losses.append(average_loss)

            end_time = time.time()
            elapsed_time = end_time - start_time
            times.append(elapsed_time)
            average_interval_time = np.array(times).mean()
            batches_left = len(loader) - (idx + 1)
            estimated_time_left = average_interval_time * (batches_left / log_interval)
            # Format time for display
            formatted_elapsed = format_time(((idx + 1) // log_interval) * elapsed_time)
            formatted_estimated = format_time(estimated_time_left)

            print(f"Epoch: {epoch}, Batch: {idx+1}/{len(loader)} | Time: {formatted_elapsed}, Est. Time Left: {formatted_estimated}")
            
            start_time = time.time()  
            total_loss = 0.0
            total_batches = 0

    # Handle any leftover batches that didn't perfectly divide by log_interval
    if total_batches > 0:
        average_loss = total_loss / total_batches
        train_losses.append(average_loss)
        print(f"Epoch: {epoch} End | {total_batches} Batches of {len(loader)} left")

        

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
filename = f"{date_time_str}, {dim}-dim, {mode}, tinystories ckpt.pt"

for epoch in range(1, args.epochs + 1):
    start_time = time.time()
    train(model, epoch, loader, train_losses, log_interval=args.log_interval)  
    end_time = time.time()
    
    val_loss = evaluate(model, epoch, val_loader)
    val_losses.append(val_loss)
    scheduler.step()
    
    if save_checkpoints:
        save_checkpoint(args, model, optimizer, model_args, train_losses, val_losses, filename, out_dir)
    print(f"Epoch {epoch} completed in {end_time - start_time:.2f} seconds")

# for epoch in range(1, args.epochs + 1):
#     start_time = time.time()
#     train_loss = train(model, epoch, small_loader)
#     train_losses.append(train_loss)
# #     print(f"Epoch: {epoch} | Training Loss: {train_loss} | Validation Loss: {val_loss}")

#     end_time = time.time()


#     if epoch % 100 == 0:
#         val_loss = evaluate(model, epoch, small_val_loader)
#         val_losses.append(val_loss)

#     if epoch % 10 == 0:
#         print(f"Epoch {epoch} completed in {end_time - start_time:.2f} seconds")

#         if save_checkpoints:
#             save_checkpoint(args, model, optimizer, model_args, train_losses, val_losses, filename, out_dir)

