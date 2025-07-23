import torch
import torch.nn as nn
from einops import rearrange

class StandardVisionTransformer(nn.Module):
    def __init__(self, embeded_dim: int, num_layers: int, num_heads: int, ffn_dim_multiply: int = 4, dropout: float = 0.1):
        super().__init__()
        
        ffn_dim = embeded_dim * ffn_dim_multiply

        self.norm1 = nn.LayerNorm(embeded_dim)
        self.attention = nn.MultiHeadedAttention(
            embeded_dim = embeded_dim,
            num_heads = num_heads,
            dropout = dropout,
            batch_first = True
        )

        self.norm2 = nn.LayerNorm(embeded_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embeded_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embeded_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass using the Pre-LN (Layer Normalization first) architecture.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embeded_dim).
        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        '''

        # --- Self-Attention Block ---
        # The residual connection creates a shortcut for the gradients
        attn_output, _ = self.attention(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_output

        # -- FFN Output ---
        ffn_output = self.ffn(self.norm2(x))
        x = x + ffn_output

        return x
    
class BaseVit(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # --- Module 1 : Embeddings ---
        self.text_embedding = nn.Embedding(
            config.tokenizer_vocab_size,
            config.model.embded_dim,
            padding_idx = config.get('pad_token_id', 0)
        )

        # 1b. Video Embedding
        # A 2D convolution acts as an efficient patch-wise liner projection
        patch_size_channels = config.model.frames_per_video * 3
        self.video_embedding = nn.Conv2d(
            in_channels = patch_size_channels,
            out_channels = config.model.embeded_dim,
            kernel_size = config.model.video_patch_size,
            stride = config.model.video_patch_size
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.model.embeded_dim))


        self.position_embedding = nn.Parameter(torch.randn(1, config.model.max_seq_len, config.model.embeded_dim))

        self.dropout = nn.Dropout(0.1)

        # --- Module 2: Transformer Body ---
        self.transformer = nn.ModuleList(
            [StandardTransformerBlock(
                embeded_dim = config.model.embeded_dim,
                num_heads = config.model.num_heads,
            )for _ in range(config.model.num_layers)]
        )

        # --- Module 3 : Prediction Head ---
        self.head_norm = nn.Linear(config.model.embeded_dim)
        self.prediction_head = nn.Linear(config.model.embeded_dim, config.data.num_answer_classes, bias = False)

        def _embed(self, video_frames, question_ids):
            """Helper function to perform the full embedding process."""
            batch_size = video_frames.shape[0]

            # Process Video : Reshape for Conv2D, embed, and then reshape to a sequence
            # [b, t, c, h, w] -> [b, t*c, h, w]
            video_reshaped = rearrange(video_frames, 'b t c h w -> b (t c) h w')
            video_patches = self.video_embedding(video_reshaped)
            video_tokens = rearrange(video_patches, 'b d ph pw -> b (ph pw) d')

            text_tokens = self.text_embedding(question_ids)

            cls_tokens = self.cls_token.expand(batch_size, -1, -1)

            x = torch.cat((cls_tokens, text_tokens, video_tokens), dim=1)

            x = x + self.position_embedding[:, :x.size(1), :]

            return self.dropout(x)
        
        def forward(self, batch: dict):
            """
            The main forward pass for the baseline model.

            Args:
                batch (dict): A dictionary containing tensors for 'video', 'question_ids', etc.

            Returns:
                torch.Tensor: The final output logits, of shape [batch_size, num_answer_classes].
                
                Note: No auxiliary loss is returned.
            """
            video_frames = batch['video']
            question_ids = batch['question_ids']

            # 1. Convert inputs to a unified sequence of embedded tokens
            x = self._embed(video_frames, question_ids)

            # 2. Pass the sequence through the Transformer layers
            for layer in self.layers:
                x = layer(x)

            # 3. Use the final state of the [CLS] token for prediction
            # Get the first token of the sequence (our CLS token)
            cls_output = x[:, 0]
            
            # Apply final normalization and the prediction head
            logits = self.prediction_head(self.head_norm(cls_output))
            
            return logits