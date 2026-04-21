# COCO Image Caption Generator (LSTM)

## Overview
This project builds an image captioning system using a CNN encoder and an LSTM decoder trained on the COCO dataset.

The model takes an image as input and generates a natural language description of its contents.

## Architecture

Image → CNN Encoder (ResNet) → Feature Vector → LSTM Decoder → Caption

- CNN extracts visual features from images
- LSTM generates captions sequentially
- Model is trained using cross-entropy loss with teacher forcing

## Dataset

We use the COCO dataset:
- ~300,000 images
- Each image has 5 captions

Sources:
- https://cocodataset.org/#home
- https://www.kaggle.com/datasets/nikhil7280/coco-image-caption

## Project Structure

- `src/data/` → preprocessing, dataset, vocabulary
- `src/models/` → embedding + LSTM model
- `src/train/` → training loop, evaluation, inference
- `data/` → raw + processed data (not tracked in git)
