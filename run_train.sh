#!/bin/bash

echo "Running train_script.py in original mode for 10 epochs..."
python train_script.py --epochs=100 --mode=original --batch_size=200 --block_size=128

echo "Running train_script.py in hyperbolic mode for 10 epochs..."
python train_script.py --epochs=100 --mode=hyperbolic --batch_size=200 --block_size=128
