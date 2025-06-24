import torch


def _build_sincos_pos_embed(num_patches, embed_dim):
    """
    Build 1D sine-cosine positional encoding.
    Shape: [1, num_patches, embed_dim]
    """
    def get_angle(pos, i, d_model):
        return pos / (10000 ** (2 * (i // 2) / d_model))

    pos = torch.arange(num_patches).unsqueeze(1)  # [num_patches, 1]
    i = torch.arange(embed_dim).unsqueeze(0)      # [1, embed_dim]
    angle_rates = get_angle(pos, i, embed_dim)    # [num_patches, embed_dim]

    pos_encoding = torch.zeros_like(angle_rates)
    pos_encoding[:, 0::2] = torch.sin(angle_rates[:, 0::2])
    pos_encoding[:, 1::2] = torch.cos(angle_rates[:, 1::2])

    return pos_encoding.unsqueeze(0)  # [1, num_patches, embed_dim]