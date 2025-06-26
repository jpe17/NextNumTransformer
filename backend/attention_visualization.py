"""
Simple Attention Visualization for Vision Transformer - Under 100 Lines!
========================================================================
"""
import torch
import torch.nn.functional as F
import cv2
import numpy as np

def get_attention_weights(model, patches, model_config):
    """Extract attention weights from model."""
    if patches.dim() == 4: patches = patches.unsqueeze(0)
    batch_size = patches.size(0)
    
    # Encoder pass
    x = patches.flatten(start_dim=2)
    x = model.patch_embedding(x) + model.encoder_pos_embedding
    for i in range(model.num_layers):
        residual = x
        x = residual + model.encoder_attention_layers[i]['W_o'](
            F.scaled_dot_product_attention(
                model.encoder_attention_layers[i]['W_q'](model.encoder_norm1_layers[i](x)).view(batch_size, -1, model.num_heads, model.head_dim).transpose(1, 2),
                model.encoder_attention_layers[i]['W_k'](model.encoder_norm1_layers[i](x)).view(batch_size, -1, model.num_heads, model.head_dim).transpose(1, 2),
                model.encoder_attention_layers[i]['W_v'](model.encoder_norm1_layers[i](x)).view(batch_size, -1, model.num_heads, model.head_dim).transpose(1, 2)
            ).transpose(1, 2).contiguous().view(batch_size, -1, model.embed_dim))
        x = x + model.encoder_ffn_layers[i](model.encoder_norm2_layers[i](x))
    encoder_output = x
    
    # Decoder pass for cross-attention
    decoder_input = torch.tensor([[10] + [12] * (model_config['max_seq_len'] - 1)], dtype=torch.long)
    y = model.token_embedding(decoder_input) + model.decoder_pos_embedding
    i = model.num_layers - 1  # Use last layer
    
    # Self-attention
    y_norm = model.decoder_norm1_layers[i](y)
    seq_len = y.size(1)
    causal_mask = ~torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    Q = model.decoder_masked_attention_layers[i]['W_q'](y_norm).view(batch_size, -1, model.num_heads, model.head_dim).transpose(1, 2)
    K = model.decoder_masked_attention_layers[i]['W_k'](y_norm).view(batch_size, -1, model.num_heads, model.head_dim).transpose(1, 2)
    V = model.decoder_masked_attention_layers[i]['W_v'](y_norm).view(batch_size, -1, model.num_heads, model.head_dim).transpose(1, 2)
    scores = (Q @ K.transpose(-2, -1)) / (model.head_dim ** 0.5)
    scores = scores.masked_fill(causal_mask == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    y = y + model.decoder_masked_attention_layers[i]['W_o']((weights @ V).transpose(1, 2).contiguous().view(batch_size, -1, model.embed_dim))
    
    # Cross-attention (what we visualize)
    y_norm = model.decoder_norm2_layers[i](y)
    Q = model.decoder_cross_attention_layers[i]['W_q'](y_norm).view(batch_size, -1, model.num_heads, model.head_dim).transpose(1, 2)
    K = model.decoder_cross_attention_layers[i]['W_k'](encoder_output).view(batch_size, -1, model.num_heads, model.head_dim).transpose(1, 2)
    scores = (Q @ K.transpose(-2, -1)) / (model.head_dim ** 0.5)
    cross_attention = F.softmax(scores, dim=-1)
    return cross_attention[0, :, 0, :].mean(dim=0).detach()  # Average across heads, first token

def create_attention_heatmap(attention_weights, canvas_shape, patch_size):
    """Convert attention weights to heatmap."""
    canvas_height, canvas_width = canvas_shape
    patches_per_row, patches_per_col = canvas_width // patch_size, canvas_height // patch_size
    grid = attention_weights.view(patches_per_col, patches_per_row).detach().numpy()
    heatmap = np.kron(grid, np.ones((patch_size, patch_size)))
    return (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

def create_real_time_attention_overlay(frame, model, model_config, crop_coords, alpha=0.4):
    """Add live attention overlay to webcam frame."""
    try:
        x1, y1, x2, y2 = crop_coords
        cropped = frame[y1:y2, x1:x2]
        
        # Preprocessing
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (model_config['canvas_width'], model_config['canvas_height']))
        normalized = resized.astype(np.float32) / 255.0
        
        # Convert to patches
        patch_size = model_config['patch_size']
        patches = []
        for i in range(0, model_config['canvas_height'], patch_size):
            for j in range(0, model_config['canvas_width'], patch_size):
                patches.append(normalized[i:i+patch_size, j:j+patch_size])
        patches = torch.tensor(patches).unsqueeze(1)
        
        # Get attention and create overlay
        with torch.no_grad():
            attention = get_attention_weights(model, patches, model_config)
            heatmap = create_attention_heatmap(attention, (model_config['canvas_height'], model_config['canvas_width']), patch_size)
        
        heatmap_resized = cv2.resize(heatmap, (x2-x1, y2-y1))
        heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        result_frame = frame.copy()
        result_frame[y1:y2, x1:x2] = cv2.addWeighted(cropped, 1-alpha, heatmap_colored, alpha, 0)
        return result_frame
    except Exception as e:
        print(f"Attention error: {e}")
        return frame

def predict_with_attention(model, patches, model_config):
    """Run prediction with attention capture."""
    from utils import predict_sequence
    predicted_digits = predict_sequence(model, patches, model_config)
    attention = get_attention_weights(model, patches, model_config)
    return predicted_digits, [attention]

def visualize_attention_analysis(image_path, predicted_digits, attention_weights, model_config):
    """Show attention analysis."""
    import matplotlib.pyplot as plt
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None or not predicted_digits or not attention_weights: return
    
    heatmap = create_attention_heatmap(attention_weights[0], (model_config['canvas_height'], model_config['canvas_width']), model_config['patch_size'])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.imshow(image, cmap='gray'); ax1.set_title(f"Original: {predicted_digits}"); ax1.axis('off')
    ax2.imshow(image, cmap='gray', alpha=0.7); ax2.imshow(heatmap, cmap='hot', alpha=0.6); ax2.set_title("Attention"); ax2.axis('off')
    plt.tight_layout(); plt.show() 