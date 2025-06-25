"""
Simple Vision Transformer Demo
============================
"""

import torch
import os
from datetime import datetime
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
    
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)
        
    # Create model
    model_config = {
        'patch_dim': 36,
        'embed_dim': 256,
        'num_patches': 100,
        'num_heads': 2,
        'num_layers': 6,
        'ffn_ratio': 2,
        'vocab_size': 13,
        'max_seq_len': 6  # max_digits + 1 for SOS
    }
    model = VisionTransformer(**model_config)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters\n")
    
    # Show tensor flow on first batch
    print("=== Tensor Demo ===")
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
    history = train_model(model, train_loader, val_loader, epochs=100, lr=0.001, artifacts_dir=artifacts_dir)
    
    # Get timestamp from training history for consistent naming
    timestamp = history['timestamp']
    run_name = history['run_name']
    
    # Save the trained model locally with timestamp
    model_path = os.path.join(artifacts_dir, f"trained_model_{timestamp}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\n💾 Model saved locally to: {model_path}")
    
    # Save model metadata with timestamp and training info
    metadata = {
        'timestamp': timestamp,
        'run_name': run_name,
        'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'patch_dim': model_config['patch_dim'],
        'embed_dim': model_config['embed_dim'],
        'num_patches': model_config['num_patches'],
        'num_classes': model_config['vocab_size'],
        'num_heads': model_config['num_heads'],
        'num_layers': model_config['num_layers'],
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'final_train_loss': history['train_losses'][-1] if history['train_losses'] else 'N/A',
        'final_val_loss': history['val_losses'][-1] if history['val_losses'] else 'N/A',
        'final_train_accuracy': history['train_accuracies'][-1] if history['train_accuracies'] else 'N/A',
        'final_val_accuracy': history['val_accuracies'][-1] if history['val_accuracies'] else 'N/A',
        'wandb_artifact_name': f"vision_transformer_model_{timestamp}"
    }
    
    metadata_path = os.path.join(artifacts_dir, f"model_metadata_{timestamp}.txt")
    with open(metadata_path, 'w') as f:
        f.write("Vision Transformer Model Metadata\n")
        f.write("=" * 35 + "\n")
        f.write(f"Training Run: {run_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Training Date: {metadata['training_date']}\n")
        f.write("-" * 35 + "\n")
        for key, value in metadata.items():
            if key not in ['timestamp', 'run_name', 'training_date']:  # Already printed above
                f.write(f"{key}: {value}\n")
        f.write("-" * 35 + "\n")
        f.write(f"Wandb Artifact: {metadata['wandb_artifact_name']}\n")
        f.write("Note: Model is also saved as wandb artifact for remote access\n")
    
    print(f"📋 Model metadata saved to: {metadata_path}")
    print(f"🏷️  Wandb artifact name: {metadata['wandb_artifact_name']}")
    print("\n✅ Training complete! Model artifacts saved locally and to wandb.")

    # --- Analysis ---
    print("\n\n=== Post-Training Analysis ===")
    analyze_training_history(history)
    show_learning_insights(history)
    analyze_sequence_predictions(model, val_loader, num_batches=5)
    show_prediction_mistakes(model, val_loader, num_mistakes=3)
    show_prediction_example(model, val_loader)


if __name__ == "__main__":
    main() 