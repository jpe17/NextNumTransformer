"""
Simple Vision Transformer for MNIST
==================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class VisionTransformer(nn.Module):
    def __init__(self, patch_dim=36, embed_dim=16, num_patches=100, 
                 num_heads=4, num_layers=3, ffn_ratio=2, 
                 vocab_size=13, max_seq_len=10):
        super().__init__()
        
        self.num_patches = num_patches
        self.embed_dim = embed_dim 
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.patch_dim = patch_dim
        self.max_seq_len = max_seq_len
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        # --- Encoder Components ---
        self.patch_embedding = nn.Linear(patch_dim, embed_dim)
        self.encoder_pos_embedding = nn.Parameter(torch.randn(1 , num_patches, embed_dim), requires_grad=True) 
        self.encoder_attention_layers = nn.ModuleList()
        self.encoder_ffn_layers = nn.ModuleList()
        self.encoder_norm1_layers = nn.ModuleList()
        self.encoder_norm2_layers = nn.ModuleList()


        for _ in range(num_layers):
            self.encoder_attention_layers.append(nn.ModuleDict({
                'W_q': nn.Linear(embed_dim, embed_dim, bias=False), 'W_k': nn.Linear(embed_dim, embed_dim, bias=False),
                'W_v': nn.Linear(embed_dim, embed_dim, bias=False), 'W_o': nn.Linear(embed_dim, embed_dim),
            }))
            self.encoder_ffn_layers.append(nn.Sequential(
                nn.Linear(embed_dim, embed_dim * ffn_ratio), nn.ReLU(),
                nn.Linear(embed_dim * ffn_ratio, embed_dim)
            ))
            self.encoder_norm1_layers.append(nn.LayerNorm(embed_dim))
            self.encoder_norm2_layers.append(nn.LayerNorm(embed_dim))

        # --- Decoder Components ---
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.decoder_pos_embedding = nn.Parameter(torch.randn(1 , max_seq_len, embed_dim), requires_grad=True) 
        self.decoder_masked_attention_layers = nn.ModuleList()
        self.decoder_cross_attention_layers = nn.ModuleList()
        self.decoder_ffn_layers = nn.ModuleList()
        self.decoder_norm1_layers = nn.ModuleList()
        self.decoder_norm2_layers = nn.ModuleList()
        self.decoder_norm3_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.decoder_masked_attention_layers.append(nn.ModuleDict({
                'W_q': nn.Linear(embed_dim, embed_dim, bias=False), 'W_k': nn.Linear(embed_dim, embed_dim, bias=False),
                'W_v': nn.Linear(embed_dim, embed_dim, bias=False), 'W_o': nn.Linear(embed_dim, embed_dim),
            }))
            self.decoder_cross_attention_layers.append(nn.ModuleDict({
                'W_q': nn.Linear(embed_dim, embed_dim, bias=False), 'W_k': nn.Linear(embed_dim, embed_dim, bias=False),
                'W_v': nn.Linear(embed_dim, embed_dim, bias=False), 'W_o': nn.Linear(embed_dim, embed_dim),
            }))
            self.decoder_ffn_layers.append(nn.Sequential(
                nn.Linear(embed_dim, embed_dim * ffn_ratio), nn.ReLU(),
                nn.Linear(embed_dim * ffn_ratio, embed_dim)
            ))
            self.decoder_norm1_layers.append(nn.LayerNorm(embed_dim))
            self.decoder_norm2_layers.append(nn.LayerNorm(embed_dim))
            self.decoder_norm3_layers.append(nn.LayerNorm(embed_dim))
            
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, src_patches, tgt_tokens):
        batch_size = src_patches.size(0)

        # --- Encoder Path ---

        x = src_patches.flatten(start_dim=2)
        x = self.patch_embedding(x)
        x = x + self.encoder_pos_embedding

        
        for i in range(self.num_layers):
            # Pre-Norm & Attention
            residual = x
            x_norm = self.encoder_norm1_layers[i](x)
            
            # Multi-Head Attention (Fixed)
            attn_module = self.encoder_attention_layers[i]
            Q = attn_module['W_q'](x_norm).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            K = attn_module['W_k'](x_norm).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            V = attn_module['W_v'](x_norm).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            
            scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
            weights = F.softmax(scores, dim=-1)
            attention_output = (weights @ V).transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
            x = residual + attn_module['W_o'](attention_output)
            
            # Pre-Norm & FFN
            residual = x
            x_norm = self.encoder_norm2_layers[i](x)
            x = residual + self.encoder_ffn_layers[i](x_norm)
            
        encoder_output = x

        # --- Decoder Path ---
        seq_len = tgt_tokens.size(1)
        y = self.token_embedding(tgt_tokens)
        y = y + self.decoder_pos_embedding
        
        # Causal Mask
        causal_mask = ~torch.triu(torch.ones(seq_len, seq_len, device=tgt_tokens.device), diagonal=1).bool()

        for i in range(self.num_layers):
            # Pre-Norm & Masked Attention
            residual = y
            y_norm = self.decoder_norm1_layers[i](y)

            # Masked Multi-Head Attention (Fixed)
            attn_module = self.decoder_masked_attention_layers[i]
            Q = attn_module['W_q'](y_norm).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            K = attn_module['W_k'](y_norm).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            V = attn_module['W_v'](y_norm).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            
            scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
            scores = scores.masked_fill(causal_mask == 0, float('-inf'))
            weights = F.softmax(scores, dim=-1)
            attention_output = (weights @ V).transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
            y = residual + attn_module['W_o'](attention_output)

            # Pre-Norm & Cross-Attention
            residual = y
            y_norm = self.decoder_norm2_layers[i](y)
            encoder_output_norm = self.encoder_norm2_layers[i](encoder_output)

            # Cross-Attention (Fixed)
            attn_module = self.decoder_cross_attention_layers[i]
            Q = attn_module['W_q'](y_norm).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            K = attn_module['W_k'](encoder_output_norm).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            V = attn_module['W_v'](encoder_output_norm).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            
            scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
            weights = F.softmax(scores, dim=-1)
            attention_output = (weights @ V).transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
            y = residual + attn_module['W_o'](attention_output)

            # Pre-Norm & FFN
            residual = y
            y_norm = self.decoder_norm3_layers[i](y)
            y = residual + self.decoder_ffn_layers[i](y_norm)
            
        return self.output_layer(y)
