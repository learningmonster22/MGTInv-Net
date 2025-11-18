import h5py
import torch
import numpy as np
from data_loader import get_magnetic_loader
from network import UNetPlusPlus
import pickle
import os

def predict_and_save(model_path, test_hdf5_path, output_hdf5_path, norm_stats_path=None):
    # --- Load normalization statistics ---
    norm_stats = None
    if norm_stats_path and os.path.exists(norm_stats_path):
        with open(norm_stats_path, 'rb') as f:
            norm_stats = pickle.load(f)
        print("Loaded normalization statistics, will use normalized data for prediction")
        normalize = True
    else:
        print("Normalization statistics file not found, will use raw data for prediction")
        normalize = False
    
    # --- Initialize model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = UNetPlusPlus(in_channels=6, out_channels=24)
    
    # Load model weights
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    print("Model loaded successfully")

    # --- Load test set (with same normalization settings) ---
    test_loader, _ = get_magnetic_loader(
        test_hdf5_path, 
        batch_size=1, 
        shuffle=False, 
        norm_stats=norm_stats, 
        normalize=normalize
    )

    if test_loader is None:
        print("Failed to load test set")
        return

    print(f"Test samples: {len(test_loader.dataset)}")

    # --- Create output HDF5 file ---
    with h5py.File(output_hdf5_path, 'w') as f_out:
        # Create prediction dataset (N, 24, 24, 24)
        predictions = f_out.create_dataset(
            "magnetic_output", 
            shape=(len(test_loader.dataset), 24, 24, 24),
            dtype=np.float32
        )

        # --- Inference and saving ---
        start_idx = 0
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(test_loader):
                inputs = inputs.to(device)
                outputs = model(inputs)
                
                # Remove channel dimension and convert to numpy array
                outputs_np = outputs.squeeze(1).cpu().numpy().astype(np.float32)  # Shape (B,24,24,24)
                batch_size = outputs_np.shape[0]
                
                # Write to corresponding position in HDF5
                predictions[start_idx:start_idx + batch_size] = outputs_np
                start_idx += batch_size
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Processed {batch_idx + 1} batches")
                    
        print(f"Prediction complete, total samples processed: {start_idx}")
        print(f"Predictions saved to: {output_hdf5_path}")

if __name__ == '__main__':
    # Configure paths
    model_path = "./results/best_model.pth"
    norm_stats_path = "./results/norm_stats.pkl"
    test_hdf5_path = "./data/test_data.h5"
    output_hdf5_path = "./results/predictions.h5"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_hdf5_path), exist_ok=True)
    
    # Execute prediction
    predict_and_save(model_path, test_hdf5_path, output_hdf5_path, norm_stats_path)