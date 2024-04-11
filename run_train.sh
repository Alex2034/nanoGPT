#!/bin/bash

echo "Running train_script.py in hyperbolic mode"
python train_script.py --epochs=800 --mode=hyperbolic --batch_size=200 --block_size=128 --gpu=

echo "Running train_script.py in original mode"
python train_script.py --epochs=800 --mode=original --batch_size=1000 --block_size=128

