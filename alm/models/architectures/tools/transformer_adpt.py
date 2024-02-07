import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from alm.models.architectures.tools.embeddings import (TimestepEmbedding,
                                                       Timesteps)

from .utils import PeriodicPositionalEncoding, init_bi_biased_mask

from typing import Optional, Tuple, Union, Callable
import math

class Transformer_Adpt(nn.Module):

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
                 max_len: int = 3000,
                 id_dim: int = 10,
                 return_intermediate_dec: bool = False,
                 require_start_token: bool = False,   
                 require_time_encoding: bool = True,              
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
        self.PPE = PeriodicPositionalEncoding(latent_dim, period = max_len)

        # temporal bias
        self.memory_bi_bias = init_bi_biased_mask(max_len) # this is for the memory bias, not all the model

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

        # motion decoder
        self.motion_decoder = nn.Linear(latent_dim, nfeats)

        # used for auto-regressive decoding
        if require_start_token:
            self.start_token = nn.Parameter(torch.randn(1, 1, latent_dim), requires_grad=True)

        # style embedding
        # self.obj_vector = nn.Linear(id_dim, latent_dim, bias=False)

        id_len = 1000
        self.id_enc = torch.zeros([id_len, latent_dim])
        position = torch.arange(0, id_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, latent_dim, 2).float() * (-math.log(10000.0) / latent_dim))
        self.id_enc[:, 0::2] = torch.sin(position * div_term)
        self.id_enc[:, 1::2] = torch.cos(position * div_term)        

        nn.init.constant_(self.motion_decoder.weight, 0)
        nn.init.constant_(self.motion_decoder.bias, 0)

    def obj_vector(self, id):
        # if id is a one-hot vector
        if id.dim() == 2: # [N, id_dim]
            id = id.argmax(dim=1)
        return self.id_enc[id] # [N, id_dim]    

    def _forward(self,
                vertice_input: torch.Tensor,
                hidden_state: torch.Tensor,
                adapter: torch.Tensor = None,
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
        
        vertice_input = self.PPE(vertice_input)

        vertice_out = self.transformer_decoder(
            tgt=vertice_input,
            memory=hidden_state,
            adapter = adapter,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask = tgt_key_padding_mask,
            memory_mask=memory_mask,
            memory_key_padding_mask = memory_key_padding_mask,
            **kwargs
        )

        vertice_out = self.motion_decoder(vertice_out)

        return vertice_out

from torch.nn.modules.transformer import _get_clones
class TransformerDecoder_w_Adapter(nn.TransformerDecoder):
    """
    A transformer decoder with adapter layer.
    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-DecoderLayer in the decoder (required).
        norm: the layer normalization component (optional).
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder_w_Adapter, self).__init__(decoder_layer, num_layers, norm)
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, 
                tgt: Tensor, 
                memory: Tensor, 
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, 
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                adapter: Optional[Tensor] = None,
        ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            adapter: the adapter for the decoder layer (optional).
        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask,
                         adapter=adapter)
            
        if self.norm is not None:
            output = self.norm(output)

        return output
        

class TransformerDecoderLayer_w_Adapter(nn.TransformerDecoderLayer):
    """
    A single layer of the transformer decoder with adapter.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: if ``True``, then the input and output tensors are provided as (batch, seq, feature). Default: ``False``.
        norm_first: if ``True``, then the input and output tensors are provided as (seq, batch, feature). Default: ``False``.
        device: the desired device of the encoder layer. Default: if ``None`` will use ``torch.device("cuda")`` if ``torch.cuda.is_available()`` else ``torch.device("cpu")``
        dtype: the desired dtype of the encoder layer. Default: if ``None`` will use ``torch.float32``
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, 
                 d_model: int, 
                 nhead: int, 
                 dim_feedforward: int = 2048, 
                 dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, 
                 batch_first: bool = False, 
                 norm_first: bool = False,
                 device=None, dtype=None) -> None:

        # folow the original transformer decoder layer
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDecoderLayer_w_Adapter, self).__init__(
            d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first, **factory_kwargs)

    def forward(self, 
                tgt: Tensor, 
                memory: Tensor, 
                tgt_mask: Optional[Tensor] = None, 
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, 
                memory_key_padding_mask: Optional[Tensor] = None,
                adapter: Optional[Tensor] = None,
        ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            adapter: the adapter for the decoder layer (optional).
        Shape:
            see the docs in Transformer class.
        """
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, adapter=adapter)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask, adapter=adapter)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, adapter=adapter))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, adapter=adapter))
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block with adapter
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], 
                  key_padding_mask: Optional[Tensor],
                  adapter: Optional[Tensor] = None,
        ) -> Tensor:
        """
        Args:
            x: [B, T, E] if batch_first else [T, B, E]
            attn_mask: [T, T]
            key_padding_mask: [B, T]
            adapter: [B, A, E] if batch_first else [A, B, E]
        Returns:
            [B, T, E] if batch_first else [T, B, E]
        """
        batch_first = self.self_attn.batch_first
        # concate adapter to key and value if it is not None
        if adapter is not None:
            x_adpt = self._concate_adapter(adapter, x, batch_first=batch_first)
        else:
            x_adpt = x

        # # original self-attention block
        # tmp = self.self_attn(x, x_adpt, x_adpt, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=True, )[1]
        # # visualize attention, use sns
        # import matplotlib.pyplot as plt
        # import seaborn as sns
        # length = 100
        # fig, ax = plt.subplots(figsize=(15, 10))
        # sns.heatmap(tmp[0, :length, :length+2].detach().cpu().numpy())
        # # save to disk
        # plt.savefig('self_attention.png')

        
        x = self.self_attn(x, x_adpt, x_adpt, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False, )[0]
        return self.dropout1(x)

    # cross-attention block with adapter
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], 
                   key_padding_mask: Optional[Tensor],
                   adapter: Optional[Tensor] = None,
        ) -> Tensor:
        """
        Args:
            x: [B, T, E] if batch_first else [T, B, E]
            mem: [B, S, E] if batch_first else [S, B, E]
            attn_mask: [T, S]
            key_padding_mask: [B, T]
            adapter: [B, A, E] if batch_first else [A, B, E]
        Returns:
            [B, T, E] if batch_first else [T, B, E]
        """

        batch_first = self.multihead_attn.batch_first
        # concate adapter to key and value if it is not None
        if adapter is not None:
            mem_adpt = self._concate_adapter(adapter, mem, batch_first=batch_first)
        else:
            mem_adpt = x

        # tmp = self.multihead_attn(x, mem_adpt, mem_adpt, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=True, )[1]
        # # visualize attention, use sns
        # import matplotlib.pyplot as plt
        # import seaborn as sns
        # length = 100
        # fig, ax = plt.subplots(figsize=(15, 10))
        # sns.heatmap(tmp[0, :length, :length+2].detach().cpu().numpy())
        # # save to disk
        # plt.savefig('cross_attention.png')

        # original cross-attention block
        x = self.multihead_attn(x, mem_adpt, mem_adpt, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False, )[0]
        return self.dropout2(x)
        
    def _concate_adapter(self, adapter: Tensor, x: Tensor, batch_first: bool = True):
        """
        concate adapter ahead of x
        Args:
            adapter: [B, A, E] if batch_first else [A, B, E]
            x: [B, T, E] if batch_first else [T, B, E]
        Returns:
            x_adapted: [B, A+T, E] if batch_first else [A+T, B, E]
        """
        if batch_first:
            x_adapted = torch.concat([adapter, x], dim=1) # [B, A, E] + [B, T, E] -> [B, A+T, E]  
        else: # batch_first
            x_adapted = torch.concat([adapter, x], dim=0) # [A, B, E] + [T, B, E] -> [A+T, B, E]
        return x_adapted



        
