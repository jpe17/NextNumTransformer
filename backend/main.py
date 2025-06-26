"""
Simple Vision Transformer Demo
============================
"""

import torch
import os
import json
from datetime import datetime
from data_processing import get_data, save_canvas_sequence
from model import VisionTransformer
from training import train_model
from torch.utils.data import TensorDataset, DataLoader
from analysis_tools import (
    analyze_training_history,
    show_learning_insights,
    analyze_sequence_predictions,
    show_prediction_mistakes,
    show_prediction_example
)
from utils import load_config, derive_model_params

config = load_config('backend/config.json')
if config is None:
    raise ValueError("Could not load config.json")

config = derive_model_params(config)


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
        num_sequences=config['num_train_sequences'], 
        max_digits=config['max_digits'],
        num_source_images=config['num_train_images']
    )
    val_canvas_patches, (val_decoder_inputs, val_target_outputs) = get_data(
        'val',
        num_sequences=config['num_val_sequences'],
        max_digits=config['max_digits'],
        num_source_images=config['num_val_images']
    )
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(train_canvas_patches, train_decoder_inputs, train_target_outputs)
    val_dataset = TensorDataset(val_canvas_patches, val_decoder_inputs, val_target_outputs)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        
    # Save a sample sequence
    save_canvas_sequence(
        train_canvas_patches, 
        (train_decoder_inputs, train_target_outputs), 
        num_to_save=30, 
        save_dir="canvas"
    )
    
    # Create model with config parameters
    model_config = {
        'patch_dim': config['patch_dim'],
        'embed_dim': config['embed_dim'],
        'num_patches': config['num_patches'],
        'num_heads': config['num_heads'],
        'num_layers': config['num_layers'],
        'ffn_ratio': config['ffn_ratio'],
        'vocab_size': config['vocab_size'],
        'max_seq_len': config['max_seq_len']
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
    history = train_model(model, train_loader, val_loader, config, epochs=config['num_epochs'], lr=config['learning_rate'], artifacts_dir=artifacts_dir)

    # --- Analysis ---
    print("\n\n=== Post-Training Analysis ===")
    analyze_training_history(history)
    show_learning_insights(history)
    analyze_sequence_predictions(model, val_loader, num_batches=5)
    show_prediction_mistakes(model, val_loader, num_mistakes=3)
    show_prediction_example(model, val_loader)


if __name__ == "__main__":
    main() 