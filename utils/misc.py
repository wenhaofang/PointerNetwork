import os
import tqdm
import numpy as np
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

    info = info_value_of_dtype(dtype)
    return info.min

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

def masked_acc(output, target, mask):
    masked_output = torch.masked_select(output, mask)
    masked_target = torch.masked_select(target, mask)
    return masked_output.eq(masked_target).float().mean()

def train(module, loader, criterion, optimizer, device, max_seq_len):
    module.train()
    epoch_loss = 0.0
    all_outputs = []
    all_targets = []
    all_maskeds = []
    for mini_batch in tqdm.tqdm(loader):
        mini_batch = [each_i.to(device) for each_i in mini_batch]
        source, target, source_length = mini_batch
        scores, output, masked_tensor = module(source, source_length)
        loss = criterion(
            scores.view(-1, scores.shape[-1]),
            target.view(-1)
        )
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_outputs.append(F.pad(output, (0, max_seq_len - output.shape[-1]), 'constant', max_seq_len))
        all_targets.append(F.pad(target, (0, max_seq_len - target.shape[-1]), 'constant', max_seq_len))
        all_maskeds.append(F.pad(masked_tensor, (0, max_seq_len - masked_tensor.shape[-1]), 'constant', False))

    all_outputs = torch.cat(all_outputs, dim = 0)
    all_targets = torch.cat(all_targets, dim = 0)
    all_maskeds = torch.cat(all_maskeds, dim = 0)

    return {
        'loss': epoch_loss / len(loader),
        'acc' : masked_acc(all_outputs, all_targets, all_maskeds),
    }

def valid(module, loader, criterion, optimizer, device, max_seq_len):
    module.eval()
    epoch_loss = 0.0
    all_outputs = []
    all_targets = []
    all_maskeds = []
    for mini_batch in tqdm.tqdm(loader):
        mini_batch = [each_i.to(device) for each_i in mini_batch]
        source, target, source_length = mini_batch
        scores, output, masked_tensor = module(source, source_length)
        loss = criterion(
            scores.view(-1, scores.shape[-1]),
            target.view(-1)
        )
        epoch_loss += loss.item()
        all_outputs.append(F.pad(output, (0, max_seq_len - output.shape[-1]), 'constant', max_seq_len))
        all_targets.append(F.pad(target, (0, max_seq_len - target.shape[-1]), 'constant', max_seq_len))
        all_maskeds.append(F.pad(masked_tensor, (0, max_seq_len - masked_tensor.shape[-1]), 'constant', False))

    all_outputs = torch.cat(all_outputs, dim = 0)
    all_targets = torch.cat(all_targets, dim = 0)
    all_maskeds = torch.cat(all_maskeds, dim = 0)

    return {
        'loss': epoch_loss / len(loader),
        'acc' : masked_acc(all_outputs, all_targets, all_maskeds),
        'outputs': all_outputs,
        'targets': all_targets,
        'maskeds': all_maskeds,
    }

def save_checkpoint(save_path, model, optim, epoch):
    checkpoint = {
        'model': model.state_dict(),
        'optim': optim.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, save_path)

def load_checkpoint(load_path, model, optim):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model'])
    optim.load_state_dict(checkpoint['optim'])
    return checkpoint['epoch']

def save_sample(folder, outputs, targets, maskeds):
    output_path = os.path.join(folder, 'outputs.txt')
    target_path = os.path.join(folder, 'targets.txt')

    outputs = outputs.cpu().numpy() if torch.is_tensor(outputs) else np.array(outputs)
    targets = targets.cpu().numpy() if torch.is_tensor(targets) else np.array(targets)
    maskeds = maskeds.cpu().numpy() if torch.is_tensor(maskeds) else np.array(maskeds)

    word_split = ' '
    line_split = '\n'

    with open(output_path, 'w', encoding = 'utf-8') as txt_file:
        for output,masked in zip(outputs , maskeds):
            masked_output = output[masked].tolist()
            masked_output = [str(i) for i in masked_output]
            cur_line = word_split.join(masked_output) + line_split
            txt_file.write(cur_line)

    with open(target_path, 'w', encoding = 'utf-8') as txt_file:
        for target,masked in zip(targets , maskeds):
            masked_target = target[masked].tolist()
            masked_target = [str(i) for i in masked_target]
            cur_line = word_split.join(masked_target) + line_split
            txt_file.write(cur_line)
