"""
Simple Vision Transformer for MNIST
==================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import _build_sincos_pos_embed


class VisionTransformerEncoder(nn.Module):
    
    def __init__(self, patch_dim=36, embed_dim=16, num_patches=100, 
                 num_heads=4, num_layers=3, ffn_ratio=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.head_dim = embed_dim // num_heads  # 16 // 4 = 4 dimensions per head
        self.num_layers = num_layers
        self.num_patches = num_patches
        
        # Patch embedding: converts flattened 6x6 patches (36 pixels) to 16-dim embedding vectors
        self.patch_embedding = nn.Linear(patch_dim, embed_dim)
        
        # Sinusoidal positional embeddings for each patch position
        self.register_buffer('pos_embedding', _build_sincos_pos_embed(num_patches, embed_dim)) # [1, 100, 16]
        
        # Multi-head attention components (we'll use these in the forward pass)
        self.W_q = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_layers)])
        self.W_k = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_layers)])
        self.W_v = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_layers)])
        self.W_o = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_layers)])
        
        # Feed-forward networks for each layer
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * ffn_ratio),  # 16 -> 32
                nn.ReLU(),
                nn.Linear(embed_dim * ffn_ratio, embed_dim)   # 32 -> 16
            ) for _ in range(num_layers)
        ])
        
        # Layer normalization for each transformer layer (2 per layer: before attention, before FFN)
        self.norm1_layers = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        self.norm2_layers = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        
        # Final classification head
        self.classifier = nn.Linear(embed_dim, num_classes)  # 32 -> 10
        
    def forward(self, x):
        # INPUT: x shape = [batch_size, 100, 1, 6, 6] - 100 patches of 6x6 pixels each
        batch_size = x.shape[0]
        
        # Step 1: Flatten patches from [batch, 100, 1, 6, 6] to [batch, 100, 36]
        x = x.flatten(start_dim=2)
        # Data shape: [batch_size, 100, 36]
        
        # Step 2: Embed patches using linear projection [batch, 100, 36] -> [batch, 100, 16]
        x = self.patch_embedding(x)
        # Data shape: [batch_size, 100, 16]
        
        # Step 3: Add positional embeddings to tell the model where each patch is located
        x = x + self.pos_embedding  # Broadcasting: [batch, 100, 16] + [1, 100, 16]
        # Data shape: [batch_size, 100, 16] - embeddings now contain positional information
        
        # Step 4: Pass through transformer layers
        for layer_idx in range(self.num_layers):
            # === MULTI-HEAD SELF-ATTENTION ===
            
            # Pre-attention layer norm
            residual = x  # Save for residual connection
            x = self.norm1_layers[layer_idx](x)
            # Data shape: [batch_size, 100, 16] - normalized embeddings
            
            # Compute Query, Key, Value matrices
            Q = self.W_q[layer_idx](x)
            K = self.W_k[layer_idx](x)
            V = self.W_v[layer_idx](x)
            
            # Reshape for multi-head attention: split embedding into num_heads
            # From [batch, 100, 16] to [batch, 4, 100, 4]
            Q = Q.view(batch_size, self.num_patches, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(batch_size, self.num_patches, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.view(batch_size, self.num_patches, self.num_heads, self.head_dim).transpose(1, 2)
            # Data shape: [batch_size, 4, 100, 4]
            
            # Scaled dot-product attention
            # Q @ K^T gives attention scores between all pairs of patches
            attention_scores = Q @ K.transpose(-2, -1)
            attention_scores = attention_scores / (self.head_dim ** 0.5)
            # Data shape: [batch_size, 4, 100, 100] - attention scores between all patch pairs
            
            # Softmax to get attention weights (probabilities)
            attention_weights = F.softmax(attention_scores, dim=-1)
            # Data shape: [batch_size, 4, 100, 100] - normalized attention weights
            
            # Apply attention weights to values
            attention_output = attention_weights @ V
            # Data shape: [batch_size, 4, 100, 4] - attended values for each head
            
            # Concatenate heads: [batch, 4, 100, 4] -> [batch, 100, 16]
            attention_output = attention_output.transpose(1, 2).contiguous()
            attention_output = attention_output.view(batch_size, self.num_patches, self.embed_dim)
            # Data shape: [batch_size, 100, 16] - concatenated multi-head attention output
            
            # Final linear projection of attention output
            x = self.W_o[layer_idx](attention_output)
            # Data shape: [batch_size, 100, 16] - projected attention output
            
            # Residual connection: add input to attention output
            x = residual + x
            # Data shape: [batch_size, 100, 16] - with residual connection
            
            # === FEED-FORWARD NETWORK ===
            
            # Pre-FFN layer norm
            residual = x  # Save for residual connection
            x = self.norm2_layers[layer_idx](x)
            # Data shape: [batch_size, 100, 16] - normalized embeddings
            
            # Feed-forward network
            x = self.ffn_layers[layer_idx](x)
            # Data shape: [batch_size, 100, 16] - FFN output
            
            # Residual connection: add input to FFN output
            x = residual + x
            # Data shape: [batch_size, 100, 16] - final layer output with residual
        
        # Step 5: The encoder's job is to return the processed sequence of patch embeddings.
        # The decoder will then use this sequence to generate the output digits.
        return x
    

class VisionTransformerDecoder(nn.Module):

