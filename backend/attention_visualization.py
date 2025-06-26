"""
Real-Time Attention Visualization for Vision Transformer
=======================================================
Shows live attention overlay during webcam scanning.
"""
import torch
import torch.nn.functional as F
import cv2
import numpy as np

# Token definitions (matching utils.py)
SOS_TOKEN = 10
EOS_TOKEN = 11  
PAD_TOKEN = 12

def predict_with_attention_capture(model, patches, model_config):
    """
    Run autoregressive prediction while capturing cross-attention weights.
    Simplified version for real-time use.
    """
    if patches.dim() == 4:
        patches = patches.unsqueeze(0)
    
    batch_size = patches.size(0)
    max_seq_len = model_config['max_seq_len']
    
    # Run encoder once
    x = patches.flatten(start_dim=2)
    x = model.patch_embedding(x) + model.encoder_pos_embedding
    
    for i in range(model.num_layers):
        residual = x
        x_norm = model.encoder_norm1_layers[i](x)
        
        attn_module = model.encoder_attention_layers[i]
        Q = attn_module['W_q'](x_norm).view(batch_size, -1, model.num_heads, model.head_dim).transpose(1, 2)
        K = attn_module['W_k'](x_norm).view(batch_size, -1, model.num_heads, model.head_dim).transpose(1, 2)
        V = attn_module['W_v'](x_norm).view(batch_size, -1, model.num_heads, model.head_dim).transpose(1, 2)
        
        scores = (Q @ K.transpose(-2, -1)) / (model.head_dim ** 0.5)
        weights = F.softmax(scores, dim=-1)
        attention_output = (weights @ V).transpose(1, 2).contiguous().view(batch_size, -1, model.embed_dim)
        x = residual + attn_module['W_o'](attention_output)
        x = x + model.encoder_ffn_layers[i](model.encoder_norm2_layers[i](x))
    
    encoder_output = x
    
    # Autoregressive generation with attention capture
    generated = [SOS_TOKEN]
    predicted_digits = []
    attention_weights = None
    
    with torch.no_grad():
        for step in range(max_seq_len - 1):
            # Prepare decoder input
            decoder_input = generated + [PAD_TOKEN] * (max_seq_len - len(generated))
            decoder_input = torch.tensor([decoder_input], dtype=torch.long)
            
            # Decoder forward pass
            y = model.token_embedding(decoder_input) + model.decoder_pos_embedding
            causal_mask = ~torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
            
            for i in range(model.num_layers):
                # Masked self-attention
                residual = y
                y_norm = model.decoder_norm1_layers[i](y)
                
                attn_module = model.decoder_masked_attention_layers[i]
                Q = attn_module['W_q'](y_norm).view(batch_size, -1, model.num_heads, model.head_dim).transpose(1, 2)
                K = attn_module['W_k'](y_norm).view(batch_size, -1, model.num_heads, model.head_dim).transpose(1, 2)
                V = attn_module['W_v'](y_norm).view(batch_size, -1, model.num_heads, model.head_dim).transpose(1, 2)
                
                scores = (Q @ K.transpose(-2, -1)) / (model.head_dim ** 0.5)
                scores = scores.masked_fill(causal_mask == 0, float('-inf'))
                weights = F.softmax(scores, dim=-1)
                attention_output = (weights @ V).transpose(1, 2).contiguous().view(batch_size, -1, model.embed_dim)
                y = residual + attn_module['W_o'](attention_output)
                
                # Cross-attention  
                residual = y
                y_norm = model.decoder_norm2_layers[i](y)
                encoder_output_norm = model.encoder_norm2_layers[i](encoder_output)
                
                attn_module = model.decoder_cross_attention_layers[i]
                Q = attn_module['W_q'](y_norm).view(batch_size, -1, model.num_heads, model.head_dim).transpose(1, 2)
                K = attn_module['W_k'](encoder_output_norm).view(batch_size, -1, model.num_heads, model.head_dim).transpose(1, 2)
                V = attn_module['W_v'](encoder_output_norm).view(batch_size, -1, model.num_heads, model.head_dim).transpose(1, 2)
                
                scores = (Q @ K.transpose(-2, -1)) / (model.head_dim ** 0.5)
                scores = scores / 1.5  # Temperature scaling for softer attention
                weights = F.softmax(scores, dim=-1)
                
                # Capture attention from last layer only (for speed)
                if i == model.num_layers - 1:
                    current_pos = len(generated) - 1
                    attention_weights = weights[0, :, current_pos, :].mean(dim=0)  # Average across heads
                
                attention_output = (weights @ V).transpose(1, 2).contiguous().view(batch_size, -1, model.embed_dim)
                y = residual + attn_module['W_o'](attention_output)
                
                # FFN
                residual = y
                y_norm = model.decoder_norm3_layers[i](y)
                y = residual + model.decoder_ffn_layers[i](y_norm)
            
            # Get next token prediction
            logits = model.output_layer(y)
            next_token = torch.argmax(logits[0, len(generated)-1]).item()
            generated.append(next_token)
            
            # Store first digit only (for real-time visualization)
            if next_token < 10 and not predicted_digits:  # Only first digit
                predicted_digits.append(next_token)
                break  # Exit early for speed
            
            if next_token == EOS_TOKEN:
                break
    
    return predicted_digits, attention_weights

def create_attention_heatmap(attention_weights, canvas_shape, patch_size):
    """
    Convert attention weights to spatial heatmap.
    Simplified version for real-time use.
    """
    canvas_height, canvas_width = canvas_shape
    patches_per_row = canvas_width // patch_size
    patches_per_col = canvas_height // patch_size
    
    # Reshape attention weights to 2D grid
    grid = attention_weights.view(patches_per_col, patches_per_row).numpy()
    
    # Upscale to match canvas size
    heatmap = np.kron(grid, np.ones((patch_size, patch_size)))
    
    # Simple robust normalization
    p5, p95 = np.percentile(heatmap, [5, 95])
    heatmap = np.clip((heatmap - p5) / (p95 - p5 + 1e-8), 0, 1)
    
    # Gamma correction for better visualization
    heatmap = np.power(heatmap, 0.8)
    
    return heatmap

def create_real_time_attention_overlay(frame, model, model_config, crop_coords, alpha=0.3):
    """
    Create real-time attention overlay on webcam frame.
    Shows attention for the first predicted digit.
    """
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
        
        # Get prediction and attention
        predicted_digits, attention_weights = predict_with_attention_capture(model, patches, model_config)
        
        if predicted_digits and attention_weights is not None:
            # Create heatmap
            heatmap = create_attention_heatmap(
                attention_weights, 
                (model_config['canvas_height'], model_config['canvas_width']), 
                patch_size
            )
            
            # Apply smoothing to reduce noise
            heatmap = cv2.GaussianBlur(heatmap, (5, 5), 1.0)
            
            # Threshold to only show strong attention areas
            threshold = np.percentile(heatmap, 70)  # Only top 30% of attention
            heatmap[heatmap < threshold] = 0
            
            # Resize heatmap to match crop size
            heatmap_resized = cv2.resize(heatmap, (x2-x1, y2-y1))
            heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_HOT)
            
            # Apply overlay
            result_frame = frame.copy()
            result_frame[y1:y2, x1:x2] = cv2.addWeighted(cropped, 1-alpha, heatmap_colored, alpha, 0)
            
            # Add prediction text
            if predicted_digits:
                digit_text = str(predicted_digits[0])
                cv2.putText(result_frame, f'Live: {digit_text}', (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            return result_frame
        
    except Exception as e:
        print(f"Attention overlay error: {e}")
    
    return frame 