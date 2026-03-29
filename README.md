📝 Project Overview

The goal of this project is to classify audio clips into one of 10 music genres:

Blues
Classical
Country
Disco
Hip-Hop
Jazz
Metal
Pop
Reggae
Rock

We experiment with three different models:

SimpleCNN – Baseline convolutional neural network.
AST Transformer – Pretrained audio spectrogram transformer fine-tuned for our dataset.
EfficientNet-B0 – Transfer learning using image classification architecture applied to spectrograms.

Each model is trained, validated, and tested separately. Ensemble predictions can further improve accuracy.

⚙️ Features
Converts audio files into Mel-spectrograms.
Implements multi-crop inference (TTA) for better predictions.
Handles NaN and Inf values in spectrogram preprocessing.
GPU-optimized training and inference pipeline.
Ready-to-use for Kaggle submission.

⚡ Usage
1. Install dependencies
pip install torch torchvision torchaudio transformers librosa opencv-python pandas numpy tqdm
2. Training
Update dataset paths in the respective training scripts (cnn_training.py, efficientnet_training.py, ast_train.py).
Run training for your desired model:
python training/cnn_training.py
3. Inference
Update paths for test data and model weights in the inference scripts.
Run inference to generate Kaggle-ready CSV:
python inference/efficientnet_inference.py
Example output:
id,genre
1,rock
2,jazz
3,metal
...
📊 Results
Model	Kaggle Score
CNN	0.736
AST Transformer	0.82
EfficientNet-B0	0.90
