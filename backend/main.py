"""
Simple Vision Transformer Demo
============================
"""

import torch
import os
from data_processing import get_data
from transformer_architecture import VisionTransformer
from training_engine import train_model
from torch.utils.data import TensorDataset, DataLoader
from analysis_tools import (
    analyze_training_history,
    show_learning_insights,
    analyze_sequence_predictions,
    show_prediction_mistakes,
    show_prediction_example
)



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
    train_canvas_patches, (train_decoder_inputs, train_target_outputs) = get_data(
        'train', 
        num_sequences=5000, 
        max_digits=5,
        num_source_images=42000
    )
    val_canvas_patches, (val_decoder_inputs, val_target_outputs) = get_data(
        'val',
        num_sequences=1000,
        max_digits=5,
        num_source_images=10000 # Use all val images
    )
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(train_canvas_patches, train_decoder_inputs, train_target_outputs)
    val_dataset = TensorDataset(val_canvas_patches, val_decoder_inputs, val_target_outputs)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
    # Create model
    model_config = {
        'patch_dim': 36,
        'embed_dim': 16,
        'num_patches': 100,
        'num_heads': 2,
        'num_layers': 2,
        'ffn_ratio': 2,
        'vocab_size': 13,
        'max_seq_len': 6  # max_digits + 1 for SOS
    }
    model = VisionTransformer(**model_config)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters\n")
    
    # Show tensor flow on first batch
    print("=== Tensor Flow Demo ===")
    sample_patches, sample_decoder_inputs, sample_target_outputs = next(iter(train_loader))
    print(f"Processing batch with {len(sample_patches)} images")
    
    with torch.no_grad():
        logits = model(sample_patches, sample_decoder_inputs)
    
    print("#"*50)
    print("Logits shape:", logits.shape) # Expected: [batch_size, seq_len, vocab_size]
    print("Target shape:", sample_target_outputs.shape) # Expected: [batch_size, seq_len]
    print("#"*50)
    
    # Train the model
    print("=== Training ===")
    history = train_model(model, train_loader, val_loader, epochs=100, lr=0.001)
    
    # Save the trained model
    model_path = os.path.join(artifacts_dir, "trained_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    # Save model metadata
    metadata = {
        'patch_dim': model_config['patch_dim'],
        'embed_dim': model_config['embed_dim'],
        'num_patches': model_config['num_patches'],
        'num_classes': model_config['vocab_size'],
        'num_heads': model_config['num_heads'],
        'num_layers': model_config['num_layers'],
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

    # --- Analysis ---
    print("\n\n=== Post-Training Analysis ===")
    analyze_training_history(history)
    show_learning_insights(history)
    analyze_sequence_predictions(model, val_loader, num_batches=5)
    show_prediction_mistakes(model, val_loader, num_mistakes=3)
    show_prediction_example(model, val_loader)


if __name__ == "__main__":
    main() 