import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import  pad_packed_sequence

from utils.misc import masked_log_softmax
from utils.misc import masked_max

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super(Attention, self).__init__()
        self.W1 = nn.Linear(hid_dim, hid_dim, bias = False)
        self.W2 = nn.Linear(hid_dim, hid_dim, bias = False)
        self.vt = nn.Linear(hid_dim, 1, bias = False)

    def forward(self, encoder_output, decoder_hidden, mask):
        '''
        Params:
            encoder_output: Torch FloatTensor (batch_size, seq_len, hid_dim)
            decoder_hidden: Torch FloatTensor (batch_size, hid_dim)
            mask          : Torch FloatTensor (batch_size, seq_len)
        Return:
            log_score     : Torch FloatTensor (batch_size, seq_len)
        '''
        encoder_transform = self.W1(encoder_output)
        decoder_transform = self.W2(decoder_hidden).unsqueeze(1)

        ori_score = self.vt(torch.tanh(encoder_transform + decoder_transform)).squeeze(-1)
        log_score = masked_log_softmax(ori_score, mask, dim = -1)

        return log_score

class PointerNetwork(nn.Module):
    def __init__(self, val_max, emb_dim, hid_dim, num_layers, num_directions, dropout):
        super(PointerNetwork, self).__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.num_directions = num_directions

        pad_idx = val_max + 1 # src_pad_idx
        emb_num = val_max + 2
        bidirectional = False if num_directions == 1 else \
                        True  if num_directions == 2 else \
                        False

        self.dropout   = nn.Dropout  (dropout)
        self.embedding = nn.Embedding(emb_num, emb_dim, padding_idx = pad_idx)
        self.encoder   = nn.LSTM     (emb_dim, hid_dim, batch_first = True, num_layers = num_layers, bidirectional = bidirectional)
        self.decoder   = nn.LSTMCell (hid_dim, hid_dim)
        self.attention = Attention   (hid_dim)

    def forward(self, src, src_len):
        '''
        Params:
            src    : Torch LongTensor (batch_size, seq_len)
            src_len: Torch LongTensor (batch_size)
        Return:
            pointer_scores: Torch FloatTensor (batch_size, seq_len, seq_len)
            pointer_indexs: Torch  LongTensor (batch_size, seq_len)
        '''
        batch_size, seq_len = src.shape[0], src.shape[1]

        embedded = self.embedding(src) # 有 dropout，会下降
        embedded = pack_padded_sequence(embedded, src_len.to('cpu'), batch_first = True, enforce_sorted = False)
        outputs, hiddens = self.encoder(embedded)
        outputs, lengths = pad_packed_sequence(outputs, batch_first = True)

        encoder_outputs = outputs
        encoder_hiddens = hiddens

        if  self.num_directions == 2:
            encoder_outputs = encoder_outputs[:, :, :self.hid_dim] + encoder_outputs[:, :, self.hid_dim:]

        encoder_h = encoder_hiddens[0]
        encoder_c = encoder_hiddens[1]
        encoder_h = encoder_h.view(self.num_layers, self.num_directions, batch_size, self.hid_dim)
        encoder_c = encoder_c.view(self.num_layers, self.num_directions, batch_size, self.hid_dim)

        if  self.num_directions == 2:
            encoder_h = encoder_h[-1, 0, :, :] + encoder_h[-1, 1, :, :]
            encoder_c = encoder_c[-1, 0, :, :] + encoder_c[-1, 1, :, :]
        else:
            encoder_h = encoder_h[-1, 0, :, :]
            encoder_c = encoder_c[-1, 0, :, :]

        decoder_hiddens = (
            encoder_h,
            encoder_c,
        )
        decoder_inputs = torch.zeros((batch_size, self.hid_dim), dtype = torch.float, device = src.device)

        arange_tensors = torch.arange(seq_len, dtype = src_len.dtype, device = src_len.device).repeat(batch_size, seq_len, 1)
        legnth_tensors = src_len.view(batch_size, 1, 1).repeat(1, seq_len, seq_len)

        row_tensors = arange_tensors < legnth_tensors
        col_tensors = row_tensors.transpose(1, 2)

        msk_tensors = row_tensors * col_tensors

        pointer_scores = []
        pointer_indexs = []
        for i in range(seq_len):
            sub_mask = msk_tensors[:, i, :]
            decoder_hiddens = self.decoder(decoder_inputs , decoder_hiddens)
            pointer_score = self.attention(encoder_outputs, decoder_hiddens[0], sub_mask)
            pointer_value , pointer_index = masked_max(pointer_score, sub_mask, dim = 1, keepdim = True)
            pointer_scores.append(pointer_score)
            pointer_indexs.append(pointer_index)
            indices_tensor = pointer_index.unsqueeze(-1).repeat( 1, 1, self.hid_dim )
            decoder_inputs = torch.gather(encoder_outputs, dim = 1, index = indices_tensor).squeeze(1)

        pointer_scores = torch.stack(pointer_scores, 1)
        pointer_indexs = torch.stack(pointer_indexs, 1).squeeze(-1)

        return pointer_scores, pointer_indexs, msk_tensors[:, 0]

def get_module(option):
    return PointerNetwork(
        option.val_max, option.emb_dim, option.hid_dim, option.num_layers, option.num_directions, option.dropout
    )

if  __name__ == '__main__':
    from utils.parser import get_parser

    parser = get_parser()
    option = parser.parse_args()

    module = get_module(option)

    src = torch.randint(option.val_min, option.val_max, (option.batch_size, option.num_max)).long()
    src_len = torch.randint(option.num_min, option.num_max + 1, (option.batch_size,)).sort(dim = 0, descending = True)[0].long()

    pointer_scores, pointer_indexs, msk_tensors = module(src, src_len)
    print(pointer_scores.shape) # (batch_size, src_len, src_len)
    print(pointer_indexs.shape) # (batch_size, src_len)
    print(msk_tensors.shape)    # (batch_size, src_len)
