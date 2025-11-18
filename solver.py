import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle

class Solver:
    """Training controller"""
    def __init__(self, model, train_loader, val_loader, test_loader=None, lr=1e-4, device='cuda', 
                 result_dir='./results', patience=20, normalize=True, norm_stats=None):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()
        self.best_loss = float('inf')
        self.result_dir = result_dir
        self.normalize = normalize
        self.norm_stats = norm_stats
        
        os.makedirs(self.result_dir, exist_ok=True)
        
        # Early stopping parameters
        self.patience = patience
        self.counter = 0
        self.best_epoch = 0
        
        # Loss records
        self.train_losses = []
        self.val_losses = []
        self.test_loss = None

    def train_epoch(self):
        """Single training epoch"""
        self.model.train()
        total_loss = 0.0
        
        for inputs, targets in tqdm(self.train_loader, desc="Training"):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Loss calculation
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # Calculate epoch average loss
        epoch_loss = total_loss / len(self.train_loader)
        self.train_losses.append(epoch_loss)
            
        return epoch_loss

    def validate(self, loader, desc="Validation"):
        """Validation process"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in tqdm(loader, desc=desc):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        
        # Calculate average loss
        epoch_loss = total_loss / len(loader)
        return epoch_loss

    def evaluate_test(self):
        """Evaluate on test set"""
        if self.test_loader is not None:
            test_loss = self.validate(self.test_loader, "Testing")
            self.test_loss = test_loss
            print(f"[Test] Final loss: {test_loss:.6f}")
            
            # Save test results
            test_result_path = os.path.join(self.result_dir, 'test_results.txt')
            with open(test_result_path, 'w') as f:
                f.write(f"Test loss: {test_loss:.6f}\n")
                f.write(f"Best model epoch: {self.best_epoch}\n")
                f.write(f"Best validation loss: {self.best_loss:.6f}\n")
            print(f"Test results saved to: {test_result_path}")

    def _plot_loss_curves(self):
        """Plot loss curves"""
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.train_losses) + 1)
        
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        # Mark best model position
        if self.best_epoch > 0:
            plt.axvline(x=self.best_epoch, color='g', linestyle='--', 
                       label=f'Best Model (Epoch {self.best_epoch})')
        
        plt.title('Training and Validation Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.result_dir, 'loss_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Loss curves saved to: {save_path}")

    def train(self, epochs):
        """Complete training process"""
        try:
            early_stop_triggered = False
            
            for epoch in range(epochs):
                train_loss = self.train_epoch()
                val_loss = self.validate(self.val_loader)
                self.val_losses.append(val_loss)
                
                # Print losses
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"[Train] Loss: {train_loss:.6f}")
                print(f"[Val]  Loss: {val_loss:.6f}")
                
                # Check if best model
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.counter = 0
                    self.best_epoch = epoch + 1
                    save_path = os.path.join(self.result_dir, "best_model.pth")
                    torch.save(self.model.state_dict(), save_path)
                    
                    # Also save normalization statistics
                    if self.normalize and self.norm_stats is not None:
                        norm_stats_path = os.path.join(self.result_dir, 'norm_stats.pkl')
                        with open(norm_stats_path, 'wb') as f:
                            pickle.dump(self.norm_stats, f)
                    
                    print(f"Model saved to {save_path}")
                else:
                    self.counter += 1
                    print(f"Validation loss not improved ({self.counter}/{self.patience})")
                
                # Check early stopping condition
                if self.counter >= self.patience:
                    early_stop_triggered = True
                    print(f"\nEarly stopping triggered! Validation loss not improved for {self.patience} epochs")
                    print(f"Best model at epoch {self.best_epoch}, validation loss: {self.best_loss:.6f}")
                    break
                
                print()
                
            # If early stopping not triggered, normal completion
            if not early_stop_triggered:
                print(f"\nTraining completed! Best model at epoch {self.best_epoch}, validation loss: {self.best_loss:.6f}")
            
            # Evaluate best model on test set
            print("\nEvaluating best model on test set...")
            self.model.load_state_dict(torch.load(os.path.join(self.result_dir, "best_model.pth")))
            self.evaluate_test()
            
            # Plot loss curves
            self._plot_loss_curves()
            
        except Exception as e:
            print(f"\nTraining terminated abnormally: {str(e)}")
            raise