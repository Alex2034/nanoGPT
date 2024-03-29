#!/bin/bash

echo "Running train_script.py in original mode for 2000 epochs..."
python train_script.py --epochs=2000 --mode=original

echo "Running train_script.py in hyperbolic mode for 2000 epochs..."
python train_script.py --epochs=2000 --mode=hyperbolic
