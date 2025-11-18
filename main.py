import argparse
import os
import pickle
from data_loader import get_magnetic_loader
from network import UNetPlusPlus
from solver import Solver
import torch

def main():
    parser = argparse.ArgumentParser(description="Train a 2D-to-3D U-Net++ network for magnetic inversion (APSTF version)")

    # --- Dataset parameters ---
    parser.add_argument('--train_path', type=str, required=True,
                        help='Path to training set HDF5 file.')
    parser.add_argument('--val_path', type=str, required=True,
                        help='Path to validation set HDF5 file.')
    parser.add_argument('--test_path', type=str, required=True,
                        help='Path to test set HDF5 file.')

    # --- Training parameters ---
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for Adam optimizer.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of DataLoader worker processes.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Training device (cuda or cpu).')
    parser.add_argument('--patience', type=int, default=20, 
                        help='Early stopping patience (epochs to wait for validation loss improvement).')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Whether to normalize data.')

    # --- Results directory ---
    parser.add_argument('--result_dir', type=str, default='./results',
                        help='Directory to save results.')

    args = parser.parse_args()
    print("Parameters:", args)

    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU.")
        args.device = 'cpu'

    # Create results directory
    os.makedirs(args.result_dir, exist_ok=True)

    # --- Data loading ---
    print("Loading training data...")
    train_loader, norm_stats = get_magnetic_loader(
        args.train_path,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        normalize=args.normalize
    )

    if train_loader is None:
        print("Failed to load training data, exiting.")
        return

    # Save normalization statistics
    if args.normalize and norm_stats is not None:
        norm_stats_path = os.path.join(args.result_dir, 'norm_stats.pkl')
        with open(norm_stats_path, 'wb') as f:
            pickle.dump(norm_stats, f)
        print(f"Normalization statistics saved to: {norm_stats_path}")

    print("Loading validation data...")
    val_loader, _ = get_magnetic_loader(
        args.val_path,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        norm_stats=norm_stats,
        normalize=args.normalize
    )

    if val_loader is None:
        print("Failed to load validation data, exiting.")
        return

    print("Loading test data...")
    test_loader, _ = get_magnetic_loader(
        args.test_path,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        norm_stats=norm_stats,
        normalize=args.normalize
    )

    if test_loader is None:
        print("Failed to load test data, exiting.")
        return

    # --- Model initialization ---
    print("Initializing model...")
    model = UNetPlusPlus(in_channels=6, out_channels=24)

    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # --- Solver initialization and training ---
    print("Initializing solver...")
    solver = Solver(
        model, 
        train_loader, 
        val_loader,
        test_loader,  
        lr=args.lr, 
        device=args.device,
        result_dir=args.result_dir,
        patience=args.patience,
        normalize=args.normalize,
        norm_stats=norm_stats
    )

    try:
        solver.train(args.epochs)
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()