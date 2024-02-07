import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from alm.models.architectures.tools.embeddings import (TimestepEmbedding,
                                                       Timesteps)

from .utils import PeriodicPositionalEncoding, init_bi_biased_mask_faceformer, init_mem_mask_faceformer

from typing import Optional, Tuple, Union, Callable
from .tools.transformer_adpt import TransformerDecoderLayer_w_Adapter, TransformerDecoder_w_Adapter

class Adpt_Bias_Denoiser(nn.Module):
    # this model is based on the trasnformer_adpt.py but with some modifications for the diffusion denoising task
    def __init__(self,
                 nfeats: int = 15069,
                 latent_dim: list = 174,
                 ff_size: int = 1024,
                 num_layers: int = 6,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 arch: str = "trans_dec",
                 audio_encoded_dim: int = 768,
                 max_len: int = 600,
                 id_dim: int = 10,
                 return_intermediate_dec: bool = False,
                 flip_sin_to_cos: bool = True,
                 freq_shift: int = 0,
                 mem_attn_scale: float = 1.0,
                 tgt_attn_scale: float = 0.1,
                 period: int = 30,
                 no_cross: bool = False,
                 **kwargs) -> None:

        super().__init__()
        self.latent_dim = latent_dim
        self.arch = arch
        self.audio_encoded_dim = audio_encoded_dim

        # audio projecter
        self.audio_feature_map = nn.Linear(audio_encoded_dim, latent_dim)

        # motion projecter
        self.vertice_map = nn.Linear(nfeats, latent_dim)

        # periodic positional encoding
        self.PPE = PeriodicPositionalEncoding(latent_dim, period = period, max_seq_len=5000) # max_seq_len can be adjusted if thit reporst an error

        # attention bias
        assert mem_attn_scale in [-1.0, 0.0, 1.0]
        self.use_mem_attn_bias = mem_attn_scale != 0.0
        self.use_tgt_attn_bias = tgt_attn_scale != 0.0
        self.memory_bi_bias = init_mem_mask_faceformer(max_len)
        
        if tgt_attn_scale < 0.0: # means we only use the causal attention
            self.target_bi_bias = init_bi_biased_mask_faceformer(num_heads, max_len, period)
            mask = torch.triu(torch.ones(max_len, max_len), diagonal=1) == 1
            self.target_bi_bias = self.target_bi_bias.masked_fill(mask, float('-inf'))
        else:
            self.target_bi_bias = init_bi_biased_mask_faceformer(num_heads, max_len, period)



        # init decoder
        decoder_layer = TransformerDecoderLayer_w_Adapter(
            d_model=latent_dim, 
            nhead=num_heads, 
            dim_feedforward=ff_size,
            dropout=dropout, 
            activation=activation, 
            norm_first=normalize_before,
            batch_first=True
        )

        self.transformer_decoder = TransformerDecoder_w_Adapter(
            decoder_layer=decoder_layer,
            num_layers=num_layers,
            )

        # used for diffusion denoising
        self.time_proj = Timesteps(
            audio_encoded_dim, 
            flip_sin_to_cos=flip_sin_to_cos, # because baseline models is trained with this
            downscale_freq_shift=freq_shift, # same as above
        )
        self.time_embedding = TimestepEmbedding(
            audio_encoded_dim,
            latent_dim * num_layers
        )
        
        # motion decoder
        self.motion_decoder = nn.Linear(latent_dim, nfeats)
        nn.init.constant_(self.motion_decoder.weight, 0)
        nn.init.constant_(self.motion_decoder.bias, 0)

        # style embedding
        self.obj_vector = nn.Embedding(id_dim, latent_dim * num_layers, )

        # whether we do not use cross attention
        self.no_cross = no_cross

    def forward(self,
                vertice_input: torch.Tensor,
                hidden_state: torch.Tensor,
                timesteps: torch.Tensor,
                adapter: torch.Tensor = None, # conditions other than the time embedding
                tgt_mask: torch.Tensor = None,
                tgt_key_padding_mask: torch.Tensor = None,
                memory_mask: torch.Tensor = None,
                memory_key_padding_mask: torch.Tensor = None,
                **kwargs):
        """
        Auto-regressive forward pass for the decoder.
        To be used during training.
        Args:
            vertice_input: [N, T, E]
            hidden_state: [N, S, E]
            adapter: [N, A, E]
            tgt_mask: [N * H, T, T]
            tgt_key_padding_mask: [N, T]
            memory_mask: [T, S]
            memory_key_padding_mask: [N, S]
        """
        
        # vertice projection
        vertice_input = self.vertice_map(vertice_input)
        vertice_input = self.PPE(vertice_input)

        # time projection
        time_emb = self.time_proj(timesteps).to(vertice_input.device)
        time_emb = self.time_embedding(time_emb).unsqueeze(1) # time_emb.shape = [N, 1, E]

        # treat the time embedding as an adapter
        if adapter is not None:
            adapter = torch.concat([adapter, time_emb], dim=1)
        else:
            adapter = time_emb

        vertice_out = vertice_input
        # split the adpater in to num_layers pieces, in order to feed them into the transformer
        adapters = adapter.split(self.latent_dim, dim=-1)

        # concat the hidden state and the vertice input
        if self.no_cross:
            hidden_len = hidden_state.shape[1]
            vertice_out = torch.cat([hidden_state, vertice_out], dim=1)
            hidden_state = torch.cat([hidden_state, hidden_state], dim=1)

        for mod,adapter in zip(self.transformer_decoder.layers, adapters):
            vertice_out = mod(
                tgt=vertice_out,
                memory=hidden_state,
                adapter=adapter,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask = tgt_key_padding_mask,
                memory_mask=memory_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                **kwargs
            )

        if self.no_cross: # remove the hidden state
            vertice_out = vertice_out[:, hidden_len:]

        if self.transformer_decoder.norm is not None:
            vertice_out = self.transformer_decoder.norm(vertice_out)

        self.transformer_decoder.layers[0].self_attn
        vertice_out = self.motion_decoder(vertice_out)

        return vertice_out





        
