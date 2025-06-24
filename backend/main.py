"""
Simple Vision Transformer Demo
============================
"""

import torch
import os
from data_processing import get_data
from transformer_architecture import VisionTransformerEncoder
from training_engine import train_model
from torch.utils.data import TensorDataset, DataLoader



def main():
    print("=== Simple Vision Transformer Demo ===\n")
    
    # Create artifacts directory if it doesn't exist
    # Get the project root directory (parent of backend)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    artifacts_dir = os.path.join(project_root, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Load data
    print("Loading and processing data...")
    train_canvas_patches, train_canvas_labels = get_data(
        'train', 
        num_sequences=5000, 
        max_digits=5,
        num_source_images=42000
    )
    val_canvas_patches, val_canvas_labels = get_data(
        'val',
        num_sequences=1000,
        max_digits=5,
        num_source_images=None # Use all val images
    )

    print(f"Train canvas patches shape: {train_canvas_patches.shape}")
    print(f"Train canvas labels shape: {train_canvas_labels.shape}")
    print(f"Validation canvas patches shape: {val_canvas_patches.shape}")
    print(f"Validation canvas labels shape: {val_canvas_labels.shape}")
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(train_canvas_patches, train_canvas_labels)
    val_dataset = TensorDataset(val_canvas_patches, val_canvas_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
    # Create model
    model = VisionTransformerEncoder(
        patch_dim=36,      # 6x6 = 36
        embed_dim=32,      # Small embedding dimension
        num_patches=100,   # 5x20grid of patches
        num_classes=10,    # MNIST digits 0-9
        num_heads=2,       # Multi-head attention
        num_layers=2       # Just 2 transformer layers
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters\n")
    
    # Show tensor flow on first batch
    print("=== Tensor Flow Demo ===")
    sample_patches, sample_labels = next(iter(train_loader))
    print(f"Processing batch with {len(sample_labels)} images")
    
    with torch.no_grad():
        x = model(sample_patches)
        print(x.shape)
        print(x)
    
    # Train the model
    print("=== Training ===")
    train_model(model, train_loader, val_loader, epochs=5, lr=0.001)
    
    # Save the trained model
    model_path = os.path.join(artifacts_dir, "trained_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    # Save model metadata
    metadata = {
        'patch_dim': 36,
        'embed_dim': 32,
        'num_patches': 100,
        'num_classes': 10,
        'num_heads': 2,
        'num_layers': 2,
        'total_parameters': sum(p.numel() for p in model.parameters())
    }
    
    metadata_path = os.path.join(artifacts_dir, "model_metadata.txt")
    with open(metadata_path, 'w') as f:
        f.write("Vision Transformer Model Metadata\n")
        f.write("=" * 35 + "\n")
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    print(f"📋 Model metadata saved to: {metadata_path}")
    print("\n✅ Training complete! Model artifacts saved.")


if __name__ == "__main__":
    main() 