import tqdm

import torch
import torch.nn
import torch.nn.functional as F

from torchvision.utils import make_grid
from torchvision.utils import save_image

import numpy as np

import matplotlib.pyplot as plt

def criterion1(x, reco_x):
    bce = F.binary_cross_entropy(reco_x, x, reduction = 'sum')
    return bce

def criterion2(x, reco_x, mean, lstd):
    bce = F.binary_cross_entropy(reco_x, x, reduction = 'sum')
    var = torch.pow(torch.exp(lstd), 2)
    kld = torch.sum(torch.log(var) + 1 - torch.pow(mean, 2) - var) * (-0.5)
    return bce + kld

def train(module, module_id, number, device, loader, criterion, optimizer):
    module.train()
    epoch_loss = 0.0
    true_images = []
    reco_images = []
    ys = []
    zs = []

    for mini_batch in tqdm.tqdm(loader):
        image , label = mini_batch
        image = image.to(device).view(image.shape[0], 1 * 28 * 28)
        label = label.to(device)

        if  module_id == 2:
            image_noises = add_noise(image)
        if  module_id == 4:
            label_onehot = to_onehot(label, number)

        if  module_id == 1:
            reco_image, z = module(image)
        if  module_id == 2:
            reco_image, z = module(image_noises)
        if  module_id == 3:
            reco_image, z, mean, lstd = module(image)
        if  module_id == 4:
            reco_image, z, mean, lstd = module(image, label_onehot)

        if  (
            module_id == 1 or
            module_id == 2
        ):
            loss = criterion(image, reco_image)
        if  (
            module_id == 3 or
            module_id == 4
        ):
            loss = criterion(image, reco_image, mean, lstd)

        epoch_loss += loss.item()
        true_images.append(image)
        reco_images.append(reco_image)

        ys.append(label)
        zs.append(z)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    true_images = torch.cat(true_images, dim = 0)
    reco_images = torch.cat(reco_images, dim = 0)
    true_images = true_images.view(true_images.shape[0], 1, 28, 28)
    reco_images = reco_images.view(reco_images.shape[0], 1, 28, 28)
    true_images = true_images[:number]
    reco_images = reco_images[:number]

    ys = torch.cat(ys, dim = 0)
    zs = torch.cat(zs, dim = 0)

    return {
        'loss': epoch_loss / len(loader),
        'true_images': true_images,
        'reco_images': reco_images,
        'ys': ys,
        'zs': zs,
    }

def valid(module, module_id, number, device):
    module.eval()
    pred_images = None
    if  module_id == 3:
        pred_images = module.predict(number, device).view(number, 1, 28, 28)
    if  module_id == 4:
        pred_images = module.predict(to_onehot(torch.arange(number), number).to(device)).view(number, 1, 28, 28)
    return {
        'pred_images': pred_images
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

def save_sample(save_path, samples):
    if  samples == None:
        return
    save_image (make_grid (samples.cpu(), nrow = 5, normalize = True).detach(), save_path)

def save_visual(save_path, ys, zs):
    ys = ys.detach().cpu().numpy()
    zs = zs.detach().cpu().numpy()
    for yt in np.unique(ys):
        plt.scatter(
            zs[ys == yt][:, 0],
            zs[ys == yt][:, 1],
            alpha = .2,
            label = yt,
        )
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def to_onehot(i , n):
    if  i.dim() == 1:
        i = i.unsqueeze(1)

    return torch.zeros((i.shape[0], n), device = i.device).scatter(1, i, 1)

def add_noise(i):
    return i + 0.1 * torch.randn_like(i)
