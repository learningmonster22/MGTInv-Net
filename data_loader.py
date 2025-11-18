import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import pickle  

class MagneticDataset(Dataset):
    """Magnetic inversion dataset loader"""
    
    def __init__(self, hdf5_path, norm_stats=None, normalize=True):
        """
        Args:
            hdf5_path: Path to HDF5 file
            norm_stats: Normalization statistics dictionary, computed automatically if None
            normalize: Whether to normalize data
        """
        # Validate file existence
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"File not found: {hdf5_path}")
            
        self.hdf5_path = hdf5_path
        self.normalize = normalize
        self.norm_stats = norm_stats
        
        # Load all data into memory at once
        with h5py.File(hdf5_path, 'r') as f:
            # Define input/output dataset names
            input_keys = ['magnetic_input_Bxx', 'magnetic_input_Bxy', 'magnetic_input_Bxz',
                         'magnetic_input_Byy', 'magnetic_input_Byz', 'magnetic_input_Bzz']
            output_key = 'magnetic_output'
            
            # Validate dataset existence
            for key in input_keys + [output_key]:
                if key not in f:
                    raise KeyError(f"Missing dataset: {key}")
                    
            # Validate shapes
            sample_input = f[input_keys[0]]
            sample_output = f[output_key]
            if sample_input.shape[1:] != (25, 25):
                raise ValueError(f"Input shape should be (N,25,25), got {sample_input.shape}")
            if sample_output.shape[1:] != (24, 24, 24):
                raise ValueError(f"Output shape should be (N,24,24,24), got {sample_output.shape}")
                
            # Record number of samples
            self.num_samples = sample_input.shape[0]
            print(f"Successfully loaded dataset, samples: {self.num_samples}")
            
            # Load all input data into memory
            print("Loading input data to memory...")
            input_arrays = []
            for key in input_keys:
                data = f[key][:].astype(np.float32)
                input_arrays.append(data)
            
            # Stack input data [N, 6, 25, 25]
            self.input_data = np.stack(input_arrays, axis=1)
            
            # Load all output data into memory [N, 24, 24, 24]
            print("Loading output data to memory...")
            self.output_data = f[output_key][:].astype(np.float32)
            
            # Add channel dimension to output [N, 1, 24, 24, 24]
            self.output_data = np.expand_dims(self.output_data, axis=1)
            
        # Data normalization
        if self.normalize:
            self._normalize_data()
            
        print("Data loading complete, starting validation...")
        # Data validation
        self._validate_data()
    
    def _normalize_data(self):
        """Normalize data"""
        if self.norm_stats is None:
            # Compute normalization statistics automatically
            print("Computing normalization statistics...")
            self.norm_stats = {
                'input_mean': np.mean(self.input_data, axis=(0, 2, 3), keepdims=True),
                'input_std': np.std(self.input_data, axis=(0, 2, 3), keepdims=True),
                'output_mean': np.mean(self.output_data),
                'output_std': np.std(self.output_data)
            }
        
        # Apply normalization
        print("Applying data normalization...")
        self.input_data = (self.input_data - self.norm_stats['input_mean']) / (self.norm_stats['input_std'] + 1e-8)
        self.output_data = (self.output_data - self.norm_stats['output_mean']) / (self.norm_stats['output_std'] + 1e-8)
        
        print(f"Input data - mean: {np.mean(self.input_data):.4f}, std: {np.std(self.input_data):.4f}")
        print(f"Output data - mean: {np.mean(self.output_data):.4f}, std: {np.std(self.output_data):.4f}")
    
    def _validate_data(self):
        """Validate all data integrity"""
        # Reasonable ranges after normalization
        input_min, input_max = -10.0, 10.0
        output_min, output_max = -10.0, 10.0
        
        # Check only first few samples for efficiency
        check_samples = min(10, self.num_samples)
        
        for i in range(check_samples):
            input_sample = self.input_data[i]  # [6, 25, 25]
            
            # Check shape, NaN/Inf for each input channel
            for ch in range(6):
                data = input_sample[ch]
                
                # 1. Shape check
                if data.shape != (25, 25):
                    raise ValueError(f"Input data {i} channel {ch} shape error, expected (25,25), got {data.shape}")
                
                # 2. NaN/Inf check
                if np.isnan(data).any():
                    raise ValueError(f"Input data {i} channel {ch} contains NaN values")
                if np.isinf(data).any():
                    raise ValueError(f"Input data {i} channel {ch} contains Inf values")
                
                # 3. Value range check (after normalization)
                if self.normalize and ((data < input_min).any() or (data > input_max).any()):
                    min_val = np.min(data)
                    max_val = np.max(data)
                    print(f"Warning: Sample {i} input channel {ch} values out of range, range=[{min_val:.2f}, {max_val:.2f}]")
            
            # Check output data
            output_sample = self.output_data[i, 0]  # [24, 24, 24]
            
            # 1. Shape check
            if output_sample.shape != (24, 24, 24):
                raise ValueError(f"Output data {i} shape error, expected (24,24,24), got {output_sample.shape}")
            
            # 2. NaN/Inf check
            if np.isnan(output_sample).any() or np.isinf(output_sample).any():
                raise ValueError(f"Output data {i} contains NaN or Inf values")
            
            # 3. Value range check (after normalization)
            if self.normalize and ((output_sample < output_min).any() or (output_sample > output_max).any()):
                min_val = np.min(output_sample)
                max_val = np.max(output_sample)
                print(f"Warning: Sample {i} output data values out of range, range=[{min_val:.2f}, {max_val:.2f}]")
        
        print(f"Data validation complete, checked {check_samples} samples")

    def __getitem__(self, idx):
        """Return data directly from memory"""
        return torch.from_numpy(self.input_data[idx]), torch.from_numpy(self.output_data[idx])

    def __len__(self):
        return self.num_samples

def get_magnetic_loader(hdf5_path, batch_size=16, shuffle=True, num_workers=4, norm_stats=None, normalize=True):
    """Create data loader"""
    try:
        dataset = MagneticDataset(hdf5_path, norm_stats=norm_stats, normalize=normalize)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
        # Return norm_stats even if not normalizing (will be None)
        return loader, dataset.norm_stats
    except Exception as e:
        print(f"Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None