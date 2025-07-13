"""
Attention Visualization Module
=============================
Real-time attention visualization for Vision Transformer inference.
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from backend.utils import predict_sequence
from backend.data_processing import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from PIL import Image

def extract_attention_weights(model, src_patches, tgt_tokens, layer_idx=-1):
    """
    Extract attention weights from the model during forward pass.
    
    Args:
        model: The Vision Transformer model
        src_patches: Input patches tensor
        tgt_tokens: Target tokens tensor
        layer_idx: Which layer to extract attention from (-1 for last layer)
    
    Returns:
        attention_weights: Tensor of attention weights
    """
    model.eval()
    
    with torch.no_grad():
        batch_size = src_patches.size(0)
        
        # --- Encoder Path ---
        x = src_patches.flatten(start_dim=2)
        x = model.patch_embedding(x)
        x = x + model.encoder_pos_embedding
        
        encoder_attention_weights = []
        
        for i in range(model.num_layers):
            # Pre-Norm & Attention
            residual = x
            x_norm = model.encoder_norm1_layers[i](x)
            
            # Multi-Head Attention
            attn_module = model.encoder_attention_layers[i]
            Q = attn_module['W_q'](x_norm).view(batch_size, -1, model.num_heads, model.head_dim).transpose(1, 2)
            K = attn_module['W_k'](x_norm).view(batch_size, -1, model.num_heads, model.head_dim).transpose(1, 2)
            V = attn_module['W_v'](x_norm).view(batch_size, -1, model.num_heads, model.head_dim).transpose(1, 2)
            
            scores = (Q @ K.transpose(-2, -1)) / (model.head_dim ** 0.5)
            weights = torch.softmax(scores / 2.0, dim=-1)  # Temperature scaling
            
            # Store attention weights
            encoder_attention_weights.append(weights)
            
            # Continue forward pass
            weights = model.dropout(weights)
            attention_output = (weights @ V).transpose(1, 2).contiguous().view(batch_size, -1, model.embed_dim)
            x = residual + attn_module['W_o'](attention_output)
            
            # Pre-Norm & FFN
            residual = x
            x_norm = model.encoder_norm2_layers[i](x)
            x = residual + model.encoder_ffn_layers[i](x_norm)
        
        # Return attention weights from specified layer
        if layer_idx == -1:
            layer_idx = len(encoder_attention_weights) - 1
        
        return encoder_attention_weights[layer_idx]

def attention_weights_to_heatmap(attention_weights, canvas_height, canvas_width, patch_size):
    """
    Convert attention weights to a spatial heatmap.
    
    Args:
        attention_weights: Tensor of attention weights [batch, heads, seq_len, seq_len]
        canvas_height: Height of the original canvas
        canvas_width: Width of the original canvas  
        patch_size: Size of each patch
    
    Returns:
        heatmap: Numpy array representing the attention heatmap
    """
    # Average across heads and take attention to first token (CLS-like behavior)
    # or average across all tokens
    weights = attention_weights[0]  # Take first batch
    weights = weights.mean(0)  # Average across heads
    
    # Take average attention across all positions
    attention_map = weights.mean(0)  # Average across query positions
    
    # Reshape to spatial dimensions
    patches_per_row = canvas_width // patch_size
    patches_per_col = canvas_height // patch_size
    
    # Ensure we have the right number of patches
    num_patches = patches_per_row * patches_per_col
    if attention_map.size(0) != num_patches:
        # Pad or truncate as needed
        if attention_map.size(0) > num_patches:
            attention_map = attention_map[:num_patches]
        else:
            pad_size = num_patches - attention_map.size(0)
            attention_map = torch.cat([attention_map, torch.zeros(pad_size)])
    
    # Reshape to 2D spatial layout
    spatial_attention = attention_map.view(patches_per_col, patches_per_row)
    
    # Upsample to original canvas size
    spatial_attention = spatial_attention.cpu().numpy()
    heatmap = cv2.resize(spatial_attention, (canvas_width, canvas_height), interpolation=cv2.INTER_LINEAR)
    
    # Normalize to 0-1 range
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    return heatmap

def create_real_time_attention_overlay(display_frame, model, model_config, crop_coords, alpha=0.4):
    """
    Create real-time attention overlay for webcam feed.
    
    Args:
        display_frame: Current webcam frame
        model: The Vision Transformer model
        model_config: Model configuration
        crop_coords: (x1, y1, x2, y2) coordinates for cropping
        alpha: Transparency of the overlay
    
    Returns:
        display_frame: Frame with attention overlay
    """
    try:
        x1, y1, x2, y2 = crop_coords
        cropped_frame = display_frame[y1:y2, x1:x2]
        
        # Convert OpenCV image to PIL Image
        cropped_pil = Image.fromarray(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
        
        # Preprocess the cropped frame
        from backend.inference import _preprocess_image
        processed_patches, steps = _preprocess_image(cropped_pil, model_config)
        
        if processed_patches is None:
            return display_frame
        
        # Create dummy target tokens for attention extraction
        dummy_tgt = torch.tensor([[SOS_TOKEN]], dtype=torch.long)
        
        # Extract attention weights
        attention_weights = extract_attention_weights(model, processed_patches.unsqueeze(0), dummy_tgt)
        
        # Convert to heatmap
        canvas_height = model_config['canvas_height']
        canvas_width = model_config['canvas_width']
        patch_size = model_config['patch_size']
        
        heatmap = attention_weights_to_heatmap(attention_weights, canvas_height, canvas_width, patch_size)
        
        # Resize heatmap to match cropped frame size
        frame_height, frame_width = cropped_frame.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (frame_width, frame_height))
        
        # Apply colormap
        colormap = cm.get_cmap('hot')
        heatmap_colored = colormap(heatmap_resized)
        heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
        
        # Blend with original frame
        cropped_with_overlay = cv2.addWeighted(cropped_frame, 1.0 - alpha, heatmap_colored, alpha, 0)
        
        # Place back into display frame
        display_frame[y1:y2, x1:x2] = cropped_with_overlay
        
        return display_frame
        
    except Exception as e:
        print(f"Error in attention overlay: {e}")
        return display_frame

def get_attention_heatmap_and_prediction(processed_image, model, model_config):
    """
    Get attention heatmap and prediction for a processed image.
    
    Args:
        processed_image: Preprocessed image (RGB numpy array)
        model: The Vision Transformer model
        model_config: Model configuration
    
    Returns:
        tuple: (heatmap, prediction) where heatmap is BGR image for cv2 display
    """
    try:
        # Convert OpenCV image to PIL Image
        processed_pil = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        
        # Preprocess the image
        from backend.inference import _preprocess_image
        processed_patches, steps = _preprocess_image(processed_pil, model_config)
        
        if processed_patches is None:
            return None, None
        
        # Get prediction
        prediction = predict_sequence(model, processed_patches.unsqueeze(0), model_config)
        
        # Filter out special tokens
        prediction = [p for p in prediction if p < 10]
        
        # Create dummy target tokens for attention extraction
        dummy_tgt = torch.tensor([[SOS_TOKEN]], dtype=torch.long)
        
        # Extract attention weights
        attention_weights = extract_attention_weights(model, processed_patches.unsqueeze(0), dummy_tgt)
        
        # Convert to heatmap
        canvas_height = model_config['canvas_height']
        canvas_width = model_config['canvas_width']
        patch_size = model_config['patch_size']
        
        heatmap = attention_weights_to_heatmap(attention_weights, canvas_height, canvas_width, patch_size)
        
        # Resize heatmap to match processed image size
        img_height, img_width = processed_image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (img_width, img_height))
        
        # Apply colormap and convert to BGR for cv2
        colormap = cm.get_cmap('hot')
        heatmap_colored = colormap(heatmap_resized)
        heatmap_bgr = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
        heatmap_bgr = cv2.cvtColor(heatmap_bgr, cv2.COLOR_RGB2BGR)
        
        return heatmap_bgr, prediction
        
    except Exception as e:
        print(f"Error in attention heatmap generation: {e}")
        return None, None

def visualize_attention_on_image(model, processed_patches, model_config, save_path=None):
    """
    Create a detailed attention visualization for analysis.
    
    Args:
        model: The Vision Transformer model
        processed_patches: Preprocessed patches
        model_config: Model configuration
        save_path: Optional path to save the visualization
    
    Returns:
        None (displays or saves the plot)
    """
    try:
        # Create dummy target tokens
        dummy_tgt = torch.tensor([[SOS_TOKEN]], dtype=torch.long)
        
        # Extract attention weights from all layers
        attention_weights_all = []
        
        # Get attention from each layer
        for layer_idx in range(model.num_layers):
            weights = extract_attention_weights(model, processed_patches, dummy_tgt, layer_idx)
            attention_weights_all.append(weights)
        
        # Create visualization
        fig, axes = plt.subplots(1, len(attention_weights_all), figsize=(15, 5))
        if len(attention_weights_all) == 1:
            axes = [axes]
        
        canvas_height = model_config['canvas_height']
        canvas_width = model_config['canvas_width']
        patch_size = model_config['patch_size']
        
        for i, attention_weights in enumerate(attention_weights_all):
            heatmap = attention_weights_to_heatmap(attention_weights, canvas_height, canvas_width, patch_size)
            
            axes[i].imshow(heatmap, cmap='hot', interpolation='bilinear')
            axes[i].set_title(f'Layer {i+1} Attention')
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Attention visualization saved to {save_path}")
        else:
            plt.show()
            
    except Exception as e:
        print(f"Error in attention visualization: {e}") 