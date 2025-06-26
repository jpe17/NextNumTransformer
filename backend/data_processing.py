"""
Simple Sequential Data Processing
=================================
"""

import torch
import random
import json
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from utils import load_config, derive_model_params

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

# --- Canvas and Sequence Generation ---

# Define special tokens for the decoder
SOS_TOKEN = config['SOS_TOKEN']
EOS_TOKEN = config['EOS_TOKEN']
PAD_TOKEN = config['PAD_TOKEN']

def create_canvas_sequences(digit_images, digit_labels, num_sequences=1000, max_digits=5, add_noise=False):
    """
    Completely revamped logic for clean images:
    1. Create a black canvas with torch.
    2. Paste original white-on-black MNIST digits onto it with random scaling and rotation.
       Overlaps are handled by taking the brightest pixel.
    3. Invert the final canvas to get black-on-white.
    4. Add Gaussian noise if specified (training only).
    5. Apply inference-style preprocessing (contrast, erosion) to simulate real-world images.
    6. Then, and only then, create patches.
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
        # 1. Pick 1-5 images
        num_digits = random.randint(1, max_digits)
        indices = random.sample(range(len(digit_images)), num_digits)
        
        # 2. Create black canvas
        canvas = torch.zeros(canvas_height, canvas_width)
        
        # 3. Paste digits with controlled, CAPTCHA-like overlap
        positions = []
        
        # Estimate the total width of the sequence to allow for more random placement
        avg_advance = 14 
        estimated_sequence_width = digit_size + (num_digits - 1) * avg_advance
        max_start_x = max(0, canvas_width - estimated_sequence_width)
        current_x = random.randint(0, max_start_x)  # Start with larger random horizontal padding

        for idx in indices:
            # Stop if the next digit would go off the canvas
            if current_x + 28 > canvas_width:
                break
                
            # Get the original digit and convert to numpy array
            digit_np = digit_images[idx, 0].numpy()
            
            if random.random() < 0.2:
                # 20% of the time, apply random rotation and scaling
                scale = random.uniform(0.5, 1.2)
                angle = random.uniform(-45, 45)
            else:
                # 80% of the time, use default scale and no rotation
                scale = 1.0
                angle = 0
            
            # Get the center of the image for rotation
            center = (digit_size // 2, digit_size // 2)
            
            # Create rotation matrix and apply transformation
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
            
            # Apply rotation and scaling
            transformed_digit = cv2.warpAffine(digit_np, rotation_matrix, (digit_size, digit_size))
            
            # Convert back to torch tensor
            digit_tensor = torch.from_numpy(transformed_digit)

            # Random vertical position, adjusted for the transformed size
            y_offset = random.randint(0, canvas_height - digit_size)
            x_pos = current_x

            # Calculate ROI size based on digit boundaries
            roi_height = min(digit_size, canvas_height - y_offset)
            roi_width = min(digit_size, canvas_width - x_pos)

            # Paste using maximum to blend white digits, creating the overlap effect
            canvas_roi = canvas[y_offset:y_offset + roi_height, x_pos:x_pos + roi_width]
            digit_roi = digit_tensor[:roi_height, :roi_width]
            canvas[y_offset:y_offset + roi_height, x_pos:x_pos + roi_width] = torch.maximum(canvas_roi, digit_roi)

            positions.append((x_pos, digit_labels[idx].item()))

            # Advance x for the next digit, ensuring partial overlap
            advance = random.randint(12, 16)  # Causes more overlap to fit in smaller canvas
            current_x += advance

        # Get labels sorted left-to-right and truncate to max_digits
        sorted_labels = [label for _, label in sorted(positions)][:max_digits]
        
        # 4. Invert canvas to get black digits on white background
        final_canvas = 1 - canvas
        
        # 5. Add Gaussian noise if specified (training only, and only for 20% of samples)
        if add_noise and random.random() < 0.2:
            noise = torch.randn_like(final_canvas) * 0.1  # Small noise std
            final_canvas = torch.clamp(final_canvas + noise, 0, 1)

        # Apply the same preprocessing as inference to bridge the domain gap
        # between synthetic training data and real-world images.
        canvas_np = final_canvas.numpy()
        
        # High-contrast normalization (mimics inference preprocessing)
        processed_canvas = np.clip(((canvas_np - 0.45) * 8) + 0.45, 0, 1)

        # Erosion to thicken digits (mimics inference preprocessing)
        kernel = np.ones((2, 2), np.uint8)
        eroded_canvas = cv2.erode(processed_canvas, kernel)
        
        # Convert back to tensor for patchifying
        processed_tensor = torch.from_numpy(eroded_canvas.astype(np.float32))

        # 6. Patch the final canvas
        patches = processed_tensor.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
        patches = patches.contiguous().view(-1, patch_size, patch_size)
        patches = patches.unsqueeze(1) # Add channel dim: [100, 1, 6, 6]

        # 7. Create decoder input and target output sequences
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
    plt.title(f'Canvas Sequence {sequence_idx}: Labels = {valid_labels}')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return valid_labels

def save_canvas_sequence(canvas_patches, canvas_labels, num_to_save=20, save_dir="saved_canvases"):
    """
    Saves a specified number of random canvas sequences from a batch to files.
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
        filename = f"canvas_seq_{seq_idx}_labels_{'_'.join(valid_labels)}.png"
        filepath = os.path.join(save_dir, filename)
        
        plt.figure(figsize=(15, 3))
        plt.imshow(full_canvas.numpy(), cmap='gray', interpolation='lanczos')
        plt.title(f'Canvas Sequence {seq_idx}: Labels = {valid_labels}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()  # Close the figure to free up memory

    print(f"✅ Saved {num_to_save} random canvases to '{save_dir}/'")

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
        img_labels = img_labels[:num_source_images]
        
    # Add noise only to training data
    add_noise = (split == 'train')
    return create_canvas_sequences(images, img_labels, num_sequences, max_digits, add_noise)
