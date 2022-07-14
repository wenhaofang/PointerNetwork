import os
import subprocess

os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

from utils.parser import get_parser
from utils.logger import get_logger

parser = get_parser()
option = parser.parse_args()

root_path = 'result'

logs_folder = os.path.join(root_path, 'logs', option.name)
save_folder = os.path.join(root_path, 'save', option.name)
sample_folder = os.path.join(root_path, 'sample', option.name)
result_folder = os.path.join(root_path, 'result', option.name)

subprocess.run('mkdir -p %s' % logs_folder, shell = True)
subprocess.run('mkdir -p %s' % save_folder, shell = True)
subprocess.run('mkdir -p %s' % sample_folder, shell = True)
subprocess.run('mkdir -p %s' % result_folder, shell = True)

logs_path = os.path.join(logs_folder, 'main.log')
save_path = os.path.join(save_folder, 'best.pth')

logger = get_logger(option.name, logs_path)

from loaders.loader1 import get_loader as get_loader1

from modules.module1 import get_module as get_module1

from utils.misc import train, valid, save_checkpoint, load_checkpoint, save_sample

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

logger.info('prepare loader')

train_loader, valid_loader, test_loader = get_loader1(option)

logger.info('prepare module')

pointer_network = get_module1(option)

pointer_network = pointer_network.to(device)

logger.info('prepare envs')

optimizer = optim.Adam(pointer_network.parameters(), lr = option.lr, weight_decay = option.wd)

criterion = nn.CrossEntropyLoss(ignore_index = option.val_max + 1)

logger.info('start training!')

best_valid_loss = float('inf')
for epoch in range(option.num_epochs):
    train_info = train(pointer_network, train_loader, criterion, optimizer, device)
    valid_info = valid(pointer_network, valid_loader, criterion, optimizer, device)
    logger.info(
        '[Epoch %d] Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f' %
        (epoch, train_info['loss'], train_info['acc'], valid_info['loss'], valid_info['acc'])
    )
    if  best_valid_loss > valid_info['loss']:
        best_valid_loss = valid_info['loss']
        save_checkpoint(save_path, pointer_network, optimizer, epoch)
        save_sample(sample_folder, valid_info['outputs'], valid_info['targets'], valid_info['maskeds'])

logger.info('start testing!')

cur_epoch = load_checkpoint(save_path, pointer_network, optimizer)

test_info = valid(pointer_network, test_loader, criterion, optimizer, device)

logger.info('Test Loss: %f, Test Acc: %f' % (test_info['loss'], test_info['acc']))

save_sample(result_folder, test_info['outputs'], test_info['targets'], valid_info['maskeds'])
