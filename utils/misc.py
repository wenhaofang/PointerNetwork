import torch
import torch.nn as nn
import torch.nn.functional as F

# tiny_value_of_dtype, info_value_of_dtype, min_value_of_dtype, masked_log_softmax, masked_max
# from https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py

def tiny_value_of_dtype(dtype):

    if  not dtype.is_floating_point:
        raise TypeError('Only supports floating point dtypes.')

    if  dtype not in [
        torch.half,
        torch.float,
        torch.double,
    ]:
        raise TypeError('Does not support dtype ' + str(dtype))
    elif dtype == torch.half:
        return 1e-4
    elif dtype == torch.float:
        return 1e-13
    elif dtype == torch.double:
        return 1e-13

def info_value_of_dtype(dtype: torch.dtype):

    if  dtype == torch.bool:
        raise TypeError('Does not support torch.bool')

    if  dtype.is_floating_point:
        return torch.finfo(dtype)
    else:
        return torch.iinfo(dtype)

def min_value_of_dtype(dtype):

    return info_value_of_dtype(dtype).min

def masked_log_softmax(vector, mask, dim = -1):
    if  mask is not None:
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        vector = vector + (mask + tiny_value_of_dtype(vector.dtype)).log()

    return F.log_softmax(vector, dim = dim)

def masked_max(vector, mask, dim, keepdim = False):
    replaced_vector = vector.masked_fill(~mask, min_value_of_dtype(vector.dtype))
    max_value, max_index = replaced_vector.max(dim = dim, keepdim = keepdim)

    return max_value, max_index
