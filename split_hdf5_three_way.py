import h5py
import numpy as np
import os
import random

def split_hdf5_three_way(input_path, train_output_path, val_output_path, test_output_path, 
                        split_ratios=(0.8, 0.1, 0.1), seed=42):
    """
    Split an HDF5 file into training, validation, and test sets with 8:1:1 ratio
    
    Args:
    input_path (str): Path to input HDF5 file
    train_output_path (str): Output path for training set HDF5 file
    val_output_path (str): Output path for validation set HDF5 file  
    test_output_path (str): Output path for test set HDF5 file
    split_ratios (tuple): Ratios for train, val, test sets, default (0.8, 0.1, 0.1)
    seed (int): Random seed for reproducibility, default 42
    """
    # Check if input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input HDF5 file not found: {input_path}")

    # Validate split ratios
    if abs(sum(split_ratios) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {sum(split_ratios)}")
    
    if len(split_ratios) != 3:
        raise ValueError("Need 3 split ratios (train, val, test)")

    # Define expected dataset keys
    input_keys = [
        'magnetic_input_Bxx', 'magnetic_input_Bxy', 'magnetic_input_Bxz',
        'magnetic_input_Byy', 'magnetic_input_Byz', 'magnetic_input_Bzz'
    ]
    output_key = 'magnetic_output'
    required_keys = input_keys + [output_key]

    print(f"Processing file: {input_path}")
    with h5py.File(input_path, 'r') as f:
        # Verify all required keys exist
        for key in required_keys:
            if key not in f:
                raise ValueError(f"Dataset '{key}' not found in {input_path}")

        # Get total samples (assuming all inputs/outputs have same first dimension)
        first_key = input_keys[0]
        total_samples = f[first_key].shape[0]
        if total_samples == 0:
            raise ValueError("Input HDF5 file contains zero samples.")
        print(f"Total samples found: {total_samples}")

        # Print each dataset shape
        for key in required_keys:
            print(f"  - Found key '{key}' with shape {f[key].shape}")

        # Create shuffled indices
        np.random.seed(seed)
        random.seed(seed)
        indices = np.arange(total_samples)
        np.random.shuffle(indices)

        # Calculate split points
        train_end = int(total_samples * split_ratios[0])
        val_end = train_end + int(total_samples * split_ratios[1])
        
        # Get shuffled indices for three datasets
        train_indices_shuffled = indices[:train_end]
        val_indices_shuffled = indices[train_end:val_end]
        test_indices_shuffled = indices[val_end:]

        # Sort indices for h5py reading efficiency
        train_indices_sorted = np.sort(train_indices_shuffled)
        val_indices_sorted = np.sort(val_indices_shuffled)
        test_indices_sorted = np.sort(test_indices_shuffled)

        print(f"Split results:")
        print(f"  Training set: {len(train_indices_sorted)} samples ({len(train_indices_sorted)/total_samples*100:.1f}%)")
        print(f"  Validation set: {len(val_indices_sorted)} samples ({len(val_indices_sorted)/total_samples*100:.1f}%)")
        print(f"  Test set: {len(test_indices_sorted)} samples ({len(test_indices_sorted)/total_samples*100:.1f}%)")

        # Create training set HDF5 file
        print(f"Creating training set: {train_output_path}")
        os.makedirs(os.path.dirname(train_output_path), exist_ok=True)
        with h5py.File(train_output_path, 'w') as train_f:
            for key in required_keys:
                data_slice = f[key][train_indices_sorted, ...]
                train_f.create_dataset(key, data=data_slice, compression="gzip")
                print(f"    - Wrote '{key}' with shape {data_slice.shape} to training file")

        # Create validation set HDF5 file
        print(f"Creating validation set: {val_output_path}")
        os.makedirs(os.path.dirname(val_output_path), exist_ok=True)
        with h5py.File(val_output_path, 'w') as val_f:
            for key in required_keys:
                data_slice = f[key][val_indices_sorted, ...]
                val_f.create_dataset(key, data=data_slice, compression="gzip")
                print(f"    - Wrote '{key}' with shape {data_slice.shape} to validation file")

        # Create test set HDF5 file
        print(f"Creating test set: {test_output_path}")
        os.makedirs(os.path.dirname(test_output_path), exist_ok=True)
        with h5py.File(test_output_path, 'w') as test_f:
            for key in required_keys:
                data_slice = f[key][test_indices_sorted, ...]
                test_f.create_dataset(key, data=data_slice, compression="gzip")
                print(f"    - Wrote '{key}' with shape {data_slice.shape} to test file")


if __name__ == '__main__':
    # Configure paths
    input_h5 = "./data/merged_dataset.h5"
    output_dir = "./data"
    
    train_h5 = os.path.join(output_dir, 'train_data.h5')
    val_h5 = os.path.join(output_dir, 'val_data.h5')
    test_h5 = os.path.join(output_dir, 'test_data.h5')
    
    split_ratios = (0.8, 0.1, 0.1)  # Train 80%, validation 10%, test 10%

    try:
        split_hdf5_three_way(input_h5, train_h5, val_h5, test_h5, split_ratios=split_ratios)
        print(f"\nDataset successfully split 8:1:1:")
        print(f"  Training data: {train_h5}")
        print(f"  Validation data: {val_h5}")
        print(f"  Test data: {test_h5}")
    except Exception as e:
        print(f"\nError during splitting: {e}")
        import traceback
        traceback.print_exc()