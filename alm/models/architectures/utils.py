import math
import torch
import torch.nn as nn

class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=600):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
# def init_biased_mask(n_head, max_seq_len, period):
#     # this code is from https://github.com/EvelynFan/FaceFormer/blob/dfaea81983665b22b99af336a80574208cfcc099/faceformer.py#L10
#     # however, the original code is not working for the case where the batch size is not 1
#     # so I modified it a little bit
#     def get_slopes(n):
#         def get_slopes_power_of_2(n):
#             start = (2**(-2**-(math.log2(n)-3)))
#             ratio = start
#             return [start*ratio**i for i in range(n)]
#         if math.log2(n).is_integer():
#             return get_slopes_power_of_2(n)                   
#         else:                                                 
#             closest_power_of_2 = 2**math.floor(math.log2(n)) 
#             return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    
#     slopes = torch.Tensor(get_slopes(n_head))
#     bias = torch.div(
#         torch.arange(
#             start=0, 
#             end=max_seq_len, 
#             step=period,
#             dtype=torch.float
#         ).unsqueeze(1).repeat(1,period).view(-1),
#         period,
#         rounding_mode='floor'
#     )
#     bias = - torch.flip(bias,dims=[0])
#     alibi = torch.zeros(max_seq_len, max_seq_len)
#     for i in range(max_seq_len):
#         alibi[i, :i+1] = bias[-(i+1):]

#     alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
#     mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
#     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#     mask = mask.unsqueeze(0) + alibi
#     return mask


# def init_bi_biased_mask_dev(n_head, max_seq_len, period):
#     # this code is from # https://github.com/tensorflow/tensor2tensor
#     # and I modified it a little bit to match the original code in  https://github.com/EvelynFan/FaceFormer/blob/dfaea81983665b22b99af336a80574208cfcc099/faceformer.py#L10
#     # such the code is working for the case where the batch size is not 1
#     def get_slopes(n):
#         def get_slopes_power_of_2(n):
#             start = (2**(-2**-(math.log2(n)-3)))
#             ratio = start
#             return [start*ratio**i for i in range(n)]
#         if math.log2(n).is_integer():
#             return get_slopes_power_of_2(n)                   
#         else:                                                 
#             closest_power_of_2 = 2**math.floor(math.log2(n)) 
#             return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]

#     slopes = torch.Tensor(get_slopes(n_head))

#     range_vec = torch.div(
#         torch.arange(
#             start=0,
#             end=max_seq_len,
#             step=period,
#             dtype=torch.float
#         ).unsqueeze(1).repeat(1,period).view(-1),
#         period,
#         rounding_mode='floor'
#     )

#     relative_matrix = range_vec[None, :] - range_vec[:, None]
#     relative_matrix[torch.where(relative_matrix > 0)] *= -1 

#     alibi = slopes.unsqueeze(1).unsqueeze(1) * relative_matrix.unsqueeze(0)
#     return alibi

def init_biased_mask(n_head, max_seq_len, period):
    # this code is from https://github.com/EvelynFan/FaceFormer/blob/dfaea81983665b22b99af336a80574208cfcc099/faceformer.py#L10
    # however, the original code is not working for the case where the batch size is not 1
    # so I modified it a little bit
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)                   
        else:                                                 
            closest_power_of_2 = 2**math.floor(math.log2(n)) 
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    
    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.div(
        torch.arange(
            start=0, 
            end=max_seq_len, 
            step=period,
            dtype=torch.float
        ).unsqueeze(1).repeat(1,period).view(-1),
        period,
        rounding_mode='floor'
    )
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]

    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask


def init_bi_biased_mask(max_seq_len, ):
    # any attention mask that is as more than 3 elements is not working

    range_vec = torch.arange(
            start=0,
            end=max_seq_len,
            dtype=torch.float
        )

    relative_matrix = range_vec[None, :] - range_vec[:, None]
    relative_matrix[torch.where(relative_matrix > 0)] *= -1 

    return relative_matrix

def init_mem_mask_faceformer(max_seq_len):
    mask = torch.ones(max_seq_len, max_seq_len)
    # set the diagonal to 0
    mask = mask.masked_fill(torch.eye(max_seq_len) == 1, 0)
    return mask    

def init_bi_biased_mask_faceformer(n_head, max_seq_len, period):
    # any attention mask that is as more than 3 elements is not working
    def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = (2**(-2**-(math.log2(n)-3)))
                ratio = start
                return [start*ratio**i for i in range(n)]
            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)                   
            else:                                                 
                closest_power_of_2 = 2**math.floor(math.log2(n)) 
                return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]

    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.div(torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1), period, rounding_mode='floor')
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
        if i+1 < max_seq_len:
            alibi[i, i+1:] = bias[-(max_seq_len-(i+1)):].flip(dims=[0])

    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)

    return alibi

