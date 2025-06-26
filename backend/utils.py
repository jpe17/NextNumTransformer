import json
import torch
import os
import glob

# Token definitions for sequence generation
SOS_TOKEN = 10  # Start of Sequence
EOS_TOKEN = 11  # End of Sequence  
PAD_TOKEN = 12  # Padding token

def load_config(filepath):
    """Loads a JSON configuration file and returns it as a dictionary."""
    try:
        with open(filepath, 'r') as f:
            config = json.load(f)
            return config
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {filepath}")
        return None

def derive_model_params(config):
    """Derive model parameters from basic config to ensure consistency."""
    canvas_height = config['canvas_height']
    canvas_width = config['canvas_width'] 
    patch_size = config['patch_size']
    
    # Derived parameters that must be consistent
    num_patches = (canvas_height // patch_size) * (canvas_width // patch_size)
    patch_dim = patch_size * patch_size  # For grayscale images
    
    # Return config with derived parameters
    derived_config = config.copy()
    derived_config['num_patches'] = num_patches
    derived_config['patch_dim'] = patch_dim
    
    return derived_config

def find_latest_run_dir(artifacts_dir=None):
    """Find the latest run directory in artifacts."""
    if artifacts_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        artifacts_dir = os.path.join(project_root, "artifacts")
    
    # Find all run directories
    run_dirs = glob.glob(os.path.join(artifacts_dir, "run_*"))
    if not run_dirs:
        raise FileNotFoundError("No trained models found! Train a model first.")
    
    # Use the most recent one
    latest_run = sorted(run_dirs)[-1]
    return latest_run, os.path.basename(latest_run)

def load_trained_model(run_folder=None, artifacts_dir=None, verbose=True):
    """Load a trained model from artifacts directory.
    
    Args:
        run_folder: Specific run folder name (e.g., 'run_25_06_25__1_54_21'). If None, uses latest.
        artifacts_dir: Path to artifacts directory. If None, uses default.
        verbose: Whether to print loading information.
    
    Returns:
        tuple: (model, model_config)
    """
    # Get project paths
    if artifacts_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        artifacts_dir = os.path.join(project_root, "artifacts")
    
    # Use specified run folder or find latest
    if run_folder is None:
        latest_run_path, run_name = find_latest_run_dir(artifacts_dir)
        if verbose:
            print(f"🤖 Loading model from: {run_name}")
    else:
        latest_run_path = os.path.join(artifacts_dir, run_folder)
        run_name = run_folder
        if verbose:
            print(f"🤖 Loading model from: {run_name}")
    
    # Load saved config
    config_file = os.path.join(latest_run_path, "model_config.json")
    model_path = os.path.join(latest_run_path, "model.pth")
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found at {config_file}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    with open(config_file, 'r') as f:
        saved_config = json.load(f)
    
    # Ensure derived parameters are consistent
    model_config = derive_model_params(saved_config)
    
    # Dynamically import the model class to avoid circular imports at the top level
    from model import VisionTransformer
    
    # Create model config for VisionTransformer constructor
    vit_config = {
        'patch_dim': model_config['patch_dim'],
        'embed_dim': model_config['embed_dim'],
        'num_patches': model_config['num_patches'],
        'num_heads': model_config['num_heads'],
        'num_layers': model_config['num_layers'],
        'ffn_ratio': model_config['ffn_ratio'],
        'vocab_size': model_config['vocab_size'],
        'max_seq_len': model_config['max_seq_len']
    }
    
    # Load model
    model = VisionTransformer(**vit_config)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    if verbose:
        print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model, model_config

def predict_sequence(model, patches, model_config):
    """Autoregressive prediction for digit sequences.
    
    Args:
        model: Trained VisionTransformer model
        patches: Input patches tensor [num_patches, channels, height, width]
        model_config: The configuration dictionary for the loaded model.
    
    Returns:
        list: Predicted digit sequence (only digits 0-9)
    """
    # Add batch dimension if needed
    if patches.dim() == 4:
        patches = patches.unsqueeze(0)
    
    generated = [SOS_TOKEN]
    max_seq_len = model_config['max_seq_len']
    
    with torch.no_grad():
        for _ in range(max_seq_len - 1):
            decoder_input = generated + [PAD_TOKEN] * (max_seq_len - len(generated))
            decoder_input = torch.tensor([decoder_input], dtype=torch.long)
            
            logits = model(patches, decoder_input)
            next_token = torch.argmax(logits[0, len(generated)-1]).item()
            
            generated.append(next_token)
            if next_token == EOS_TOKEN:
                break
    
    # Return only digits (0-9)
    return [token for token in generated[1:] if token < 10]