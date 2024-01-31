from typing import Dict, List

import numpy as np
import torch
from torch import Tensor

def lengths_to_mask(lengths: Tensor, # [batch_size]
                    device: torch.device,
                    max_len: int = None) -> Tensor:
    max_len = max_len if max_len else max(lengths.cpu().tolist())
    mask = torch.arange(max_len, device=device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask

def remove_padding(tensors, lengths):
    return [
        tensor[:tensor_length]
        for tensor, tensor_length in zip(tensors, lengths)
    ]