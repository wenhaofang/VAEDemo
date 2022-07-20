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

from loaders.loader1 import get_loader as get_loader1 # MNIST

from modules.module1 import get_module as get_module1 # AE
from modules.module3 import get_module as get_module3 # VAE
from modules.module4 import get_module as get_module4 # CVAE

from utils.misc import train, valid, save_checkpoint, load_checkpoint, save_sample, save_visual

from utils.misc import criterion1 # BCE
from utils.misc import criterion2 # BCE + KLD

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

logger.info('prepare loader')

loader = get_loader1(option)

logger.info('prepare module')

module = get_module1(option).to(device) if option.module == 1 else \
         get_module3(option).to(device) if option.module == 3 else \
         get_module4(option).to(device) if option.module == 4 else \
         None

logger.info('prepare envs')

optimizer = optim.Adam(module.parameters())

criterion = criterion1 if option.module == 1 else \
            criterion2 if option.module == 3 else \
            criterion2 if option.module == 4 else \
            None

logger.info('start training!')

for epoch in range(1, 1 + option.num_epochs):
    train_info = train(module, option.module, option.num_labels, device, loader, criterion, optimizer)
    valid_info = valid(module, option.module, option.num_labels, device)
    logger.info('[Epoch %d] Train Loss: %f' % (epoch, train_info['loss']))
    if  epoch % option.print_freq == 0:
        save_sample(os.path.join(  sample_folder, str(epoch) + '_true.png'), train_info['true_images'])
        save_sample(os.path.join(  sample_folder, str(epoch) + '_reco.png'), train_info['reco_images'])
        save_checkpoint(os.path.join(save_folder, str(epoch) + '.pth'), module, optimizer, epoch)
        if  (
            option.module == 1 or
            option.module == 2
        ):
            save_visual(os.path.join(result_folder, str(epoch) + '_visu.png'), train_info['ys'], train_info['zs'])
        if  (
            option.module == 3 or
            option.module == 4
        ):
            save_sample(os.path.join(result_folder, str(epoch) + '_pred.png'), valid_info['pred_images'])
