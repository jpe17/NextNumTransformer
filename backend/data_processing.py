"""
Simple Sequential Data Processing
=================================
"""

import torch
import random
import json
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from utils import load_config

# Example usage:

config = load_config('backend/config.json')

# --- Data Loading and Splitting (run once) ---

# Load the full MNIST dataset
dataset = MNIST(root='.w3-data/', download=True, train=True, transform=transforms.ToTensor())
loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
ALL_IMAGES, ALL_LABELS = next(iter(loader))

# Define train/val/test splits on the original digit images
n_total = len(ALL_IMAGES)
train_end = int(n_total * config.get('train_split'))
val_end = int(n_total * config.get('val_split'))

DATA_SPLITS = {
    'train': (ALL_IMAGES[:train_end], ALL_LABELS[:train_end]),
    'val': (ALL_IMAGES[train_end:val_end], ALL_LABELS[train_end:val_end]),
    'test': (ALL_IMAGES[val_end:], ALL_LABELS[val_end:])
}

# --- Canvas and Sequence Generation ---

# Define special tokens for the decoder
SOS_TOKEN = config.get('SOS_TOKEN')  # Start of Sequence
EOS_TOKEN = config.get('EOS_TOKEN')  # End of Sequence
PAD_TOKEN = config.get('PAD_TOKEN')  # Padding token

def create_canvas_sequences(digit_images, digit_labels, num_sequences=1000, max_digits=5):
    """
    Completely revamped logic for clean images:
    1. Create a black canvas with torch.
    2. Paste original white-on-black MNIST digits onto it.
       Overlaps are handled by taking the brightest pixel.
    3. Invert the final canvas to get black-on-white.
    4. Then, and only then, create patches.
    """
    all_canvas_patches = []
    all_decoder_inputs = []
    all_target_outputs = []
    
    canvas_height, canvas_width = 30, 120
    patch_size = 6  # 30/6=5, 120/6=20 -> 5x20=100 patches
    max_seq_len = max_digits + 1  # Sequence length is max digits + 1 for SOS/EOS token

    for _ in range(num_sequences):
        # 1. Pick 1-5 images
        num_digits = random.randint(1, max_digits)
        indices = random.sample(range(len(digit_images)), num_digits)
        
        # 2. Create black canvas
        canvas = torch.zeros(canvas_height, canvas_width)
        
        # 3. Paste digits with controlled, CAPTCHA-like overlap
        positions = []
        current_x = random.randint(0, 5)  # Start with small random horizontal padding

        for idx in indices:
            # Stop if the next digit would go off the canvas
            if current_x + 28 > canvas_width:
                break
                
            digit_tensor = digit_images[idx, 0]  # Original white-on-black

            # Random vertical position
            y_offset = random.randint(0, canvas_height - 28)
            x_pos = current_x

            # Paste using maximum to blend white digits, creating the overlap effect
            canvas_roi = canvas[y_offset:y_offset + 28, x_pos:x_pos + 28]
            canvas[y_offset:y_offset + 28, x_pos:x_pos + 28] = torch.maximum(canvas_roi, digit_tensor)

            positions.append((x_pos, digit_labels[idx].item()))

            # Advance x for the next digit, ensuring partial overlap (advance < 28px)
            advance = random.randint(14, 22)  # Causes 6px to 14px of overlap
            current_x += advance

        # Get labels sorted left-to-right and truncate to max_digits
        sorted_labels = [label for _, label in sorted(positions)][:max_digits]
        
        # 4. Invert canvas to get black digits on white background
        final_canvas = 1 - canvas

        # 5. Patch the final canvas
        patches = final_canvas.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
        # patches shape: [5, 20, 6, 6]
        patches = patches.contiguous().view(-1, patch_size, patch_size)
        # patches shape: [100, 6, 6]
        patches = patches.unsqueeze(1) # Add channel dim: [100, 1, 6, 6]

        # 6. Create decoder input and target output sequences
        
        # Decoder Input: Starts with SOS, ends with PAD tokens
        # Example: [10, 8, 3, 1, 12, 12] for max_digits=5
        decoder_input = [SOS_TOKEN] + sorted_labels
        decoder_input.extend([PAD_TOKEN] * (max_seq_len - len(decoder_input)))

        # Target Output: Ends with EOS, ends with PAD tokens
        # Example: [8, 3, 1, 11, 12, 12] for max_digits=5
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


def visualize_canvas_sequence(canvas_patches, canvas_labels, sequence_idx=0):
    """Reconstructs the canvas from patches and displays it."""
    patches = canvas_patches[sequence_idx]
    
    # Check if labels are in the new tuple format (decoder_input, target_output)
    # and use the target_output for visualization.
    if isinstance(canvas_labels, tuple):
        labels = canvas_labels[1][sequence_idx] # Use target_output for labels
    else:
        labels = canvas_labels[sequence_idx]

    # Filter out special tokens (SOS, EOS, PAD) to get the true digit labels
    valid_labels = [label for label in labels.tolist() if label < 10]

    # Reconstruct the 30x120 canvas from 100 patches of 6x6
    canvas_height, canvas_width = 30, 120
    patch_size = 6
    patches_per_row = canvas_width // patch_size
    
    full_canvas = torch.zeros(canvas_height, canvas_width)
    for i, patch in enumerate(patches):
        row = i // patches_per_row
        col = i % patches_per_row
        full_canvas[row*patch_size:(row+1)*patch_size, col*patch_size:(col+1)*patch_size] = patch.squeeze()

    # Show the canvas with a smoother interpolation
    plt.figure(figsize=(15, 3))
    plt.imshow(full_canvas.numpy(), cmap='gray', interpolation='lanczos')
    plt.title(f'Canvas Sequence {sequence_idx}: Labels = {valid_labels}')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return valid_labels


def get_data(split='train', num_sequences=1000, max_digits=5, num_source_images=None):
    """
    Generates canvas sequences for a specified data split.

    Args:
        split (str): The data split to use ('train', 'val', or 'test').
        num_sequences (int): The number of canvas sequences to generate.
        max_digits (int): The maximum number of digits to place on a canvas.
        num_source_images (int, optional): The number of original digit images
            to use as a source for creating canvases. If None, all images in the
            specified split are used.

    Returns:
        A tuple of (canvas_patches, canvas_labels).
    """
    # Get the pre-split source images and labels
    images, img_labels = DATA_SPLITS[split]
    
    # Optionally, use a subset of the source images
    if num_source_images is not None:
        images = images[:num_source_images]
        img_labels = img_labels[
            :num_source_images]
        
    return create_canvas_sequences(images, img_labels, num_sequences, max_digits)
