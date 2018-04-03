#!/bin/sh

echo "Installing dependencies"
pip3 install -r requirements.txt

echo "Run training"
python3 train.py
