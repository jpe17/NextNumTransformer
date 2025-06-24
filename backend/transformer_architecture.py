"""
Simple Vision Transformer for MNIST
==================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionTransformerEncoder(nn.Module):
    
    from utils import _build_sincos_pos_embed
    
    def __init__(self, patch_dim=100, embed_dim=32, num_patches=16, num_classes=10, 
                 num_heads=4, num_layers=3, ffn_ratio=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 32 // 4 = 8 dimensions per head
        self.num_layers = num_layers
        self.num_patches = num_patches
        
        # Patch embedding: converts flattened 7x7 patches (49 pixels) to embedding vectors
        self.patch_embedding = nn.Linear(patch_dim, embed_dim)  # 49 -> 32
        
        # Learnable positional embeddings for each patch position
        self.register_buffer('pos_embedding', self._build_sincos_pos_embed(num_patches, embed_dim)) # [1, 16, 32]
        
        # Multi-head attention components (we'll use these in the forward pass)
        self.W_q = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_layers)])
        self.W_k = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_layers)])
        self.W_v = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_layers)])
        self.W_o = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_layers)])
        
        # Feed-forward networks for each layer
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * ffn_ratio),  # 32 -> 64
                nn.ReLU(),
                nn.Linear(embed_dim * ffn_ratio, embed_dim)   # 64 -> 32
            ) for _ in range(num_layers)
        ])
        
        # Layer normalization for each transformer layer (2 per layer: before attention, before FFN)
        self.norm1_layers = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        self.norm2_layers = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        
        # Final classification head
        self.classifier = nn.Linear(embed_dim, num_classes)  # 32 -> 10
        
    def forward(self, x):
        # INPUT: x shape = [batch_size, 16, 1, 7, 7] - 16 patches of 7x7 pixels each
        batch_size = x.shape[0]
        
        # Step 1: Flatten patches from [batch, num_patches, 1, 6, 6] to [batch, num_patches, 36]
        x = x.flatten(start_dim=2)  # Flatten the 1x6x6 = 36 pixels per patch
        # Data shape: [batch_size, num_patches, 36] - e.g. 100 patches, each with 36 pixel values
        
        # Step 2: Embed patches using linear projection [batch, num_patches, 36] -> [batch, num_patches, 32]
        x = self.patch_embedding(x)  # Convert 36-dim pixel vectors to 32-dim embeddings
        # Data shape: [batch_size, num_patches, 32] - e.g. 100 patch embeddings of 32 dimensions each
        
        # Step 3: Add positional embeddings to tell the model where each patch is located
        x = x + self.pos_embedding  # Broadcasting: [batch, num_patches, 32] + [1, num_patches, 32]
        # Data shape: [batch_size, num_patches, 32] - embeddings now contain positional information
        
        # Step 4: Pass through transformer layers
        for layer_idx in range(self.num_layers):
            # === MULTI-HEAD SELF-ATTENTION ===
            
            # Pre-attention layer norm
            residual = x  # Save for residual connection
            x = self.norm1_layers[layer_idx](x)  # Normalize before attention
            # Data shape: [batch_size, num_patches, 32] - normalized embeddings
            
            # Compute Query, Key, Value matrices
            Q = self.W_q[layer_idx](x)  # [batch, num_patches, 32] -> [batch, num_patches, 32]
            K = self.W_k[layer_idx](x)  # [batch, num_patches, 32] -> [batch, num_patches, 32]
            V = self.W_v[layer_idx](x)  # [batch, num_patches, 32] -> [batch, num_patches, 32]
            
            # Reshape for multi-head attention: split embedding into num_heads
            # From [batch, num_patches, 32] to [batch, num_heads, num_patches, 8]
            Q = Q.view(batch_size, self.num_patches, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(batch_size, self.num_patches, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.view(batch_size, self.num_patches, self.num_heads, self.head_dim).transpose(1, 2)
            # Data shape: [batch_size, num_heads, num_patches, 8] - e.g. 4 attention heads, 100 patches, 8 dims per head
            
            # Scaled dot-product attention
            # Q @ K^T gives attention scores between all pairs of patches
            attention_scores = Q @ K.transpose(-2, -1)  # [batch, num_heads, num_patches, num_patches]
            attention_scores = attention_scores / (self.head_dim ** 0.5)  # Scale by sqrt(head_dim)
            # Data shape: [batch_size, num_heads, num_patches, num_patches] - attention scores between all patch pairs
            
            # Softmax to get attention weights (probabilities)
            attention_weights = F.softmax(attention_scores, dim=-1)
            # Data shape: [batch_size, num_heads, num_patches, num_patches] - normalized attention weights
            
            # Apply attention weights to values
            attention_output = attention_weights @ V  # [batch, num_heads, num_patches, num_patches] @ [batch, num_heads, num_patches, 8]
            # Data shape: [batch_size, num_heads, num_patches, 8] - attended values for each head
            
            # Concatenate heads: [batch, num_heads, num_patches, 8] -> [batch, num_patches, 32]
            attention_output = attention_output.transpose(1, 2).contiguous()
            attention_output = attention_output.view(batch_size, self.num_patches, self.embed_dim)
            # Data shape: [batch_size, num_patches, 32] - concatenated multi-head attention output
            
            # Final linear projection of attention output
            x = self.W_o[layer_idx](attention_output)
            # Data shape: [batch_size, num_patches, 32] - projected attention output
            
            # Residual connection: add input to attention output
            x = residual + x
            # Data shape: [batch_size, num_patches, 32] - with residual connection
            
            # === FEED-FORWARD NETWORK ===
            
            # Pre-FFN layer norm
            residual = x  # Save for residual connection
            x = self.norm2_layers[layer_idx](x)  # Normalize before FFN
            # Data shape: [batch_size, num_patches, 32] - normalized embeddings
            
            # Feed-forward network: 32 -> 64 -> 32 (with ReLU in between)
            x = self.ffn_layers[layer_idx](x)
            # Data shape: [batch_size, num_patches, 32] - FFN output
            
            # Residual connection: add input to FFN output
            x = residual + x
            # Data shape: [batch_size, num_patches, 32] - final layer output with residual
        
        # Step 5: Global average pooling - average across all patches
        x = x.mean(dim=1)  # Average over the num_patches dimension
        # Data shape: [batch_size, 32] - single embedding vector per image
        
        # Step 6: Classification head - convert to class probabilities
        logits = self.classifier(x)  # [batch, 32] -> [batch, 10]
        # Data shape: [batch_size, 10] - logits for 10 digit classes (0-9)
        
        return logits 
    

