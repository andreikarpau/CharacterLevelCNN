#!/bin/sh

echo "Installing dependencies"
pip3 install -r requirements_train.txt

echo "Run training"
python3 train.py
