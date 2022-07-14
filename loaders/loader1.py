import torch
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class NumberSort(Dataset):
    def __init__(self, val_min, val_max, num_min, num_max, samples):
        super(NumberSort, self).__init__()

        assert val_min >= 0
        assert val_max >= val_min
        assert num_min >= 1
        assert num_max >= num_min

        self.srcs = [np.random.randint(val_min, val_max + 1, np.random.randint(num_min, num_max + 1)) for _ in range(samples)]
        self.trgs = [np.argsort(src) for src in self.srcs]

    def __getitem__(self, index):
        src = self.srcs[index].tolist()
        trg = self.trgs[index].tolist()

        return src, trg

    def __len__(self):
        assert len(self.srcs) == len(self.trgs)
        return len(self.srcs)

def collate_fn(batch_data, pad_idx):
    origin_srcs, origin_trgs = zip(*batch_data)
    sorted_srcs, sorted_trgs = zip(*sorted(zip(origin_srcs, origin_trgs), key = lambda x: len(x[0]), reverse = True))

    padded_srcs = [src + [pad_idx] * (len(sorted_srcs[0]) - len(src)) for src in sorted_srcs]
    padded_trgs = [trg + [pad_idx] * (len(sorted_trgs[0]) - len(trg)) for trg in sorted_trgs]
    src_lengths = [len(src) for src in sorted_srcs]

    return (
        torch.LongTensor(padded_srcs),
        torch.LongTensor(padded_trgs),
        torch.LongTensor(src_lengths),
    )

def get_loader(option):
    train_dataset = NumberSort(option.val_min, option.val_max, option.num_min, option.num_max, option.train_samples)
    valid_dataset = NumberSort(option.val_min, option.val_max, option.num_min, option.num_max, option.valid_samples)
    test_dataset  = NumberSort(option.val_min, option.val_max, option.num_min, option.num_max, option.test_samples )

    train_dataloader = DataLoader(train_dataset, batch_size = option.batch_size, shuffle = True , collate_fn = lambda x: collate_fn(x, option.val_max + 1))
    valid_dataloader = DataLoader(valid_dataset, batch_size = option.batch_size, shuffle = False, collate_fn = lambda x: collate_fn(x, option.val_max + 1))
    test_dataloader  = DataLoader(test_dataset , batch_size = option.batch_size, shuffle = False, collate_fn = lambda x: collate_fn(x, option.val_max + 1))

    return train_dataloader, valid_dataloader, test_dataloader

if  __name__ == '__main__':
    from utils.parser import get_parser

    parser = get_parser()
    option = parser.parse_args()

    train_loader, valid_loader, test_loader = get_loader(option)

    for mini_batch in train_loader:
        src, trg, src_len = mini_batch
        print(src.shape)     # (batch_size, seq_len)
        print(trg.shape)     # (batch_size, seq_len)
        print(src_len.shape) # (batch_size)
        break
