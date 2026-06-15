#!/usr/bin/env bash
# Run this in a Colab/Kaggle cell to train on GPU. Code stays in git; notebook is a launcher.
set -e
git clone https://github.com/<your-username>/pidetect.git
cd pidetect
pip install -q -r requirements.txt
python -m src.pidetect.data.download
# python -m src.pidetect.detect.train   # uncomment once train.py is implemented
