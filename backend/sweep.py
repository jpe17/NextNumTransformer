"""
Super Modular Wandb Sweep Runner
===============================
Just run this file to start a sweep with config_sweep.json parameters!
"""

import json
import torch
import wandb
import sys
import os
from torch.utils.data import TensorDataset, DataLoader

# Add project root to Python path for consistent imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.data_processing import get_data
from backend.model import VisionTransformer
from backend.utils import load_config, derive_model_params

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_sweep_config(filepath='backend/config_sweep.json'):
    """Load the sweep configuration."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Sweep config not found at {filepath}")
        return None


def train_sweep():
    """Training function that wandb will call for each sweep run."""
    # Initialize wandb run
    wandb.init()
    
    # Load base config
    base_config = load_config('backend/config.json')
    if base_config is None:
        raise ValueError("Could not load base config.json")
    
    # Override base config with sweep parameters
    for param, value in wandb.config.items():
        base_config[param] = value
    
    # Derive model parameters
    config = derive_model_params(base_config)
    
    # Load data
    print(f"🔄 Loading data with config: lr={config['learning_rate']}, embed_dim={config['embed_dim']}, heads={config['num_heads']}")
    
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
    
    # Create model
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
    
    # Simple training loop (no artifacts saving)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=config['PAD_TOKEN'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    for epoch in range(config['num_epochs']):
        # Training
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for patches, decoder_inputs, target_outputs in train_loader:
            optimizer.zero_grad()
            logits = model(patches, decoder_inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), target_outputs.view(-1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            mask = target_outputs != config['PAD_TOKEN']
            predicted = logits.argmax(dim=-1)
            train_correct += (predicted[mask] == target_outputs[mask]).sum().item()
            train_total += mask.sum().item()
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for patches, decoder_inputs, target_outputs in val_loader:
                logits = model(patches, decoder_inputs)
                loss = criterion(logits.view(-1, logits.size(-1)), target_outputs.view(-1))
                
                val_loss += loss.item()
                mask = target_outputs != config['PAD_TOKEN']
                predicted = logits.argmax(dim=-1)
                val_correct += (predicted[mask] == target_outputs[mask]).sum().item()
                val_total += mask.sum().item()
        
        # Calculate metrics
        train_acc = 100 * train_correct / train_total if train_total > 0 else 0
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc
        })
        
        print(f"Epoch {epoch+1}: Train {avg_train_loss:.4f} ({train_acc:.1f}%) | Val {avg_val_loss:.4f} ({val_acc:.1f}%)")


def main():
    """Main function to start the sweep."""
    print("🚀 Starting Wandb Sweep...")
    
    # Load sweep configuration
    sweep_config = load_sweep_config()
    if sweep_config is None:
        return
    
    print("📋 Sweep Configuration:")
    print(f"  Method: {sweep_config['method']}")
    print(f"  Metric: {sweep_config['metric']['name']} ({sweep_config['metric']['goal']})")
    print(f"  Parameters: {len(sweep_config['parameters'])} hyperparameters")
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, project="vision-transformer-sweep")
    print(f"🔥 Sweep created: {sweep_id}")
    
    # Start sweep agent
    print("🤖 Starting sweep agent...")
    wandb.agent(sweep_id, train_sweep)


if __name__ == "__main__":
    main() 