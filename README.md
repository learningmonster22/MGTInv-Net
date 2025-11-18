MGTInv-Net
A deep learning framework for 2D-to-3D magnetic inversion using U-Net++ architecture with Adaptive Parameterized Soft Thresholding Function (APSTF).

Requirements
The requirements.txt file contains:
txt
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
h5py>=3.3.0
tqdm>=4.62.0
matplotlib>=3.3.0
scipy>=1.7.0
scikit-learn>=0.24.0

Quick Start
1. Prepare Your Data
bash
python split_hdf5_three_way.py

2. Train the Model
bash
python main.py \
    --train_path ./data/train_data.h5 \
    --val_path ./data/val_data.h5 \
    --test_path ./data/test_data.h5
    
3. Generate Predictions
bash
python predict.py

Dataset Preparation
Input Format
The system expects HDF5 files with the following structure:

Input Components (each 25×25):
magnetic_input_Bxx,magnetic_input_Bxy,magnetic_input_Bxz,magnetic_input_Byy,magnetic_input_Byz,magnetic_input_Bzz,
Output (24×24×24):
magnetic_output

Data Splitting
Use the provided utility to split your dataset:

bash
python split_hdf5_three_way.py \
    --input_path ./data/your_dataset.h5 \
    --train_output ./data/train_data.h5 \
    --val_output ./data/val_data.h5 \
    --test_output ./data/test_data.h5
Default Split Ratio: 80% training, 10% validation, 10% test

Training
Basic Training
bash
python main.py \
    --train_path ./data/train_data.h5 \
    --val_path ./data/val_data.h5 \
    --test_path ./data/test_data.h5 \
    --result_dir ./results
Advanced Training
bash
python main.py \
    --train_path ./data/train_data.h5 \
    --val_path ./data/val_data.h5 \
    --test_path ./data/test_data.h5 \
    --batch_size 32 \
    --epochs 200 \
    --lr 0.0001 \
    --patience 30 \
    --device cuda \
    --result_dir ./experiment_1

Model Architecture
U-Net++ with APSTF
Input: 6-channel 25×25 magnetic tensor components
Encoder: 4-level convolutional downsampling [32, 64, 128, 256]
Bottleneck: 512 features with max pooling
Decoder: 4-level upsampling with dense skip connections
Output: 24-channel 24×24 features reshaped to 24×24×24

APSTF Activation Function
Adaptive Thresholding: Learns channel-wise thresholds dynamically
ECA Attention: Efficient Channel Attention for scaling factors
Soft Thresholding: Replaces traditional ReLU activation
Feature Enhancement: Preserves important features while suppressing noise

Configuration
Data Configuration (data_loader.py)
Input shape: 6×25×25
Output shape: 1×24×24×24
Automatic normalization with statistics preservation
Memory-efficient loading with validation checks

Model Configuration (network.py)
Feature channels: [32, 64, 128, 256]
APSTF activation in all convolutional blocks
Dense connections between encoder and decoder
Final sigmoid activation for output normalization

Training Configuration (solver.py)
Loss function: Mean Squared Error (MSE)
Optimizer: Adam with configurable learning rate
Early stopping based on validation loss
Automatic model checkpointing

Results
Training Outputs
best_model.pth: Best performing model weights
norm_stats.pkl: Dataset normalization statistics
loss_curves.png: Training and validation loss history
test_results.txt: Quantitative evaluation on test set

Example Output
Test loss: 0.004567
Best model epoch: 85
Best validation loss: 0.003892

License
This project is licensed under the MIT License - see the LICENSE file for details.
