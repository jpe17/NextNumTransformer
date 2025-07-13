"""
Simple Sequential Data Processing
=================================
"""

import torch
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

# Add project root to Python path for consistent imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.utils import load_config, derive_model_params

# Load config and derive parameters
# The config path is now relative to the project root where scripts are run
config = load_config('backend/config.json')
if config is None:
    raise ValueError("Could not load config.json")

config = derive_model_params(config)

# --- Data Loading and Splitting (run once) ---

# Load the full MNIST dataset
dataset = MNIST(root='.w3-data/', download=True, train=True, transform=transforms.ToTensor())
loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
ALL_IMAGES, ALL_LABELS = next(iter(loader))

# Define train/val/test splits on the original digit images
n_total = len(ALL_IMAGES)
train_end = int(n_total * config['train_split'])
val_end = int(n_total * config['val_split'])

DATA_SPLITS = {
    'train': (ALL_IMAGES[:train_end], ALL_LABELS[:train_end]),
    'val': (ALL_IMAGES[train_end:val_end], ALL_LABELS[train_end:val_end]),
    'test': (ALL_IMAGES[val_end:], ALL_LABELS[val_end:])
}

# Define special tokens for the decoder
SOS_TOKEN = config['SOS_TOKEN']
EOS_TOKEN = config['EOS_TOKEN']
PAD_TOKEN = config['PAD_TOKEN']

def create_sequences(digit_images, digit_labels, num_sequences=1000, max_digits=5):
    """
    Creates sequences of digits by placing them on a fixed-size canvas.
    This is a simplified version that places digits horizontally without complex transformations.
    """
    all_canvas_patches = []
    all_decoder_inputs = []
    all_target_outputs = []
    
    canvas_height = config['canvas_height']
    canvas_width = config['canvas_width']
    patch_size = config['patch_size']
    max_seq_len = max_digits + 1  # Sequence length is max digits + 1 for SOS/EOS token
    digit_size = 28  # Original MNIST digit size

    for _ in range(num_sequences):
        num_digits = random.randint(1, max_digits)
        indices = random.sample(range(len(digit_images)), num_digits)
        
        # Create a white canvas (background)
        canvas = torch.ones(canvas_height, canvas_width)
        
        # --- Simplified Digit Placement ---
        positions = []
        
        # Leave a small random padding at the start
        current_x = random.randint(2, 10)

        for idx in indices:
            # Stop if the next digit would go off the canvas
            if current_x + digit_size > canvas_width:
                break
                
            digit_tensor = digit_images[idx, 0] # It's already a tensor (black bg)

            # Invert digit to be black on white background before placing
            digit_tensor = 1 - digit_tensor
            
            # Random vertical position
            y_offset = random.randint(0, canvas_height - digit_size)
            x_pos = current_x

            # Paste the digit onto the canvas
            # Use minimum to blend black digits onto the white canvas
            canvas_roi = canvas[y_offset:y_offset + digit_size, x_pos:x_pos + digit_size]
            canvas[y_offset:y_offset + digit_size, x_pos:x_pos + digit_size] = torch.minimum(canvas_roi, digit_tensor)

            positions.append((x_pos, digit_labels[idx].item()))

            # Advance x for the next digit with a small random gap
            current_x += digit_size + random.randint(-5, 5)

        # Get labels sorted left-to-right
        sorted_labels = [label for _, label in sorted(positions)]
        
        # --- Patching and Sequence Creation (same as before) ---
        
        # Patch the final canvas
        patches = canvas.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
        patches = patches.contiguous().view(-1, patch_size, patch_size)
        patches = patches.unsqueeze(1) # Add channel dim

        # Create decoder input and target output sequences
        decoder_input = [SOS_TOKEN] + sorted_labels
        decoder_input.extend([PAD_TOKEN] * (max_seq_len - len(decoder_input)))

        target_output = sorted_labels + [EOS_TOKEN]
        target_output.extend([PAD_TOKEN] * (max_seq_len - len(target_output)))
        
        all_canvas_patches.append(patches)
        all_decoder_inputs.append(decoder_input)
        all_target_outputs.append(target_output)
        
    # Return patches and a tuple containing the two label tensors
    return (
        torch.stack(all_canvas_patches), 
        (torch.tensor(all_decoder_inputs, dtype=torch.long), torch.tensor(all_target_outputs, dtype=torch.long))
    )


def visualize_sequence(canvas_patches, canvas_labels, sequence_idx=0):
    """Reconstructs the image from patches and displays it."""
    patches = canvas_patches[sequence_idx]
    
    # Check if labels are in the new tuple format (decoder_input, target_output)
    # and use the target_output for visualization.
    if isinstance(canvas_labels, tuple):
        labels = canvas_labels[1][sequence_idx] # Use target_output for labels
    else:
        labels = canvas_labels[sequence_idx]

    # Filter out special tokens (SOS, EOS, PAD) to get the true digit labels
    valid_labels = [label for label in labels.tolist() if label < 10]

    # Reconstruct the canvas from patches
    canvas_height = config['canvas_height']
    canvas_width = config['canvas_width']
    patch_size = config['patch_size']
    patches_per_row = canvas_width // patch_size
    
    full_canvas = torch.zeros(canvas_height, canvas_width)
    for i, patch in enumerate(patches):
        row = i // patches_per_row
        col = i % patches_per_row
        full_canvas[row*patch_size:(row+1)*patch_size, col*patch_size:(col+1)*patch_size] = patch.squeeze()

    # Show the canvas with a smoother interpolation
    plt.figure(figsize=(15, 3))
    plt.imshow(full_canvas.numpy(), cmap='gray', interpolation='lanczos')
    plt.title(f'Sequence {sequence_idx}: Labels = {valid_labels}')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return valid_labels

def save_sequence(canvas_patches, canvas_labels, num_to_save=20, save_dir="saved_images"):
    """
    Saves a specified number of random image sequences from a batch to files.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    num_available = canvas_patches.shape[0]
    if num_to_save > num_available:
        print(f"Warning: Requested to save {num_to_save}, but only {num_available} are available. Saving all.")
        num_to_save = num_available
        
    # Select random indices to save
    indices_to_save = random.sample(range(num_available), num_to_save)

    canvas_height = config['canvas_height']
    canvas_width = config['canvas_width']
    patch_size = config['patch_size']
    patches_per_row = canvas_width // patch_size

    for seq_idx in indices_to_save:
        patches = canvas_patches[seq_idx]
        
        if isinstance(canvas_labels, tuple):
            labels = canvas_labels[1][seq_idx]  # Use target_output for labels
        else:
            labels = canvas_labels[seq_idx]
            
        # Convert labels to strings for the filename
        valid_labels = [str(label.item()) for label in labels if label < 10]
        
        # Reconstruct the canvas
        full_canvas = torch.zeros(canvas_height, canvas_width)
        for i, patch in enumerate(patches):
            row = i // patches_per_row
            col = i % patches_per_row
            full_canvas[row*patch_size:(row+1)*patch_size, col*patch_size:(col+1)*patch_size] = patch.squeeze()

        # Save the figure
        filename = f"sequence_{seq_idx}_labels_{'_'.join(valid_labels)}.png"
        filepath = os.path.join(save_dir, filename)
        
        plt.figure(figsize=(15, 3))
        plt.imshow(full_canvas.numpy(), cmap='gray', interpolation='lanczos')
        plt.title(f'Sequence {seq_idx}: Labels = {valid_labels}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()  # Close the figure to free up memory

    print(f"✅ Saved {num_to_save} random images to '{save_dir}/'")

def get_data(split='train', num_sequences=1000, max_digits=5, num_source_images=None):
    """
    Generates image sequences for a specified data split.

    Args:
        split (str): The data split to use ('train', 'val', or 'test').
        num_sequences (int): The number of image sequences to generate.
        max_digits (int): The maximum number of digits to place on an image.
        num_source_images (int, optional): The number of original digit images
            to use as a source for creating sequences. If None, all images in the
            specified split are used.

    Returns:
        A tuple of (image_patches, image_labels).
    """
    # Get the pre-split source images and labels
    images, img_labels = DATA_SPLITS[split]
    
    # Optionally, use a subset of the source images
    if num_source_images is not None:
        images = images[:num_source_images]
        img_labels = img_labels[:num_source_images]
        
    return create_sequences(images, img_labels, num_sequences, max_digits)


