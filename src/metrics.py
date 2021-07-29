import torch
import torchvision
import torchaudio
import numpy
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

#Vector-Wise

def cosine_similarity(x,y):
    # x, y = x.detach().cpu().numpy(), y.detach().cpu().numpy()
    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)
    cos = np.dot(x, y)/(nx * ny)
    return min(max(cos, -1), 1)

def angular_similarity(x,y):
    cos = cosine_similarity(x, y)
    return 1 - np.arccos(cos)/np.pi

def kl_divergence(x, y):
    denom_bound = 0.1
    return sum(x[i] * np.log(x[i]/(x[i]+denom_bound)) for i in range(x.size(0)))

def hypersphere_norm(x):
    return torch.nn.functional.normalize(x, p=2, dim=-1)

def l2_distance(x,y):
    return torch.cdist(x, y, p=2)


#Batch-Wise
def instance_loss(x):
    sim_matrix = torch.triu(torch.mm(x, x.t()), diagonal=1)
    # sim_matrix[torch.eye(len(x)).byte()] = 0
    loss = sim_matrix.mean()
    return loss

def nce_loss(x, y, temp=1, mask=None):
    sim_matrix = temp * x @ y.T
    pos_pairs = torch.arange(x.size(0)).to(sim_matrix.device)
    loss =  torch.nn.functional.cross_entropy(sim_matrix, pos_pairs) 
    loss += torch.nn.functional.cross_entropy(sim_matrix.T, pos_pairs)
    loss *= 0.5
    return loss

def centroid_loss(x, y, z, mode="nce", mask=None):
    #compute centroid as arithmetic mean of latent vectors x, y, z
    if mode=="positive":
        return centroid_positive_loss(x,y,z)
    if mode=="contrastive":
        return centroid_contrastive_loss(x,y,z)
    if mode=="nce":
        return centroid_nce_loss(x,y,z, mask)        
    else:
        return

def centroid_positive_loss(x, y, z):
    centroid = (x + y + z)/3
    x_loss = l2_distance(x, centroid).mean()
    y_loss = l2_distance(y, centroid).mean()
    z_loss = l2_distance(z, centroid).mean()
    positive_loss = x_loss + y_loss + z_loss
    return positive_loss

def centroid_contrastive_loss(x, y, z):
    centroid = (x + y + z)/3
    x_loss = l2_distance(x, centroid).mean()
    y_loss = l2_distance(y, centroid).mean()
    z_loss = l2_distance(z, centroid).mean()
    positive_loss = x_loss + y_loss + z_loss

    negative_loss = 0
    for i in range(1, len(centroid)):
        for k in range(i):
            negative_loss += cosine_similarity(centroid[i], centroid[k])

    bsz = x.shape[0]
    neg_loss_weight = 2/(bsz**2)
    pos_loss_weight = 1/bsz
    loss_scaling = 1

    loss = (neg_loss_weight * negative_loss) + (pos_loss_weight * positive_loss)
    return loss_scaling * loss

def centroid_nce_loss(x, y, z, mask=None):
    centroid = (x + y + z)/3
    nce_losses = 0
    # nce_losses += 0.5*(nce_loss(x, centroid) + nce_loss(centroid, x)) 
    # nce_losses += 0.5*(nce_loss(y, centroid) + nce_loss(centroid, y))
    # nce_losses += 0.5*(nce_loss(z, centroid) + nce_loss(centroid, z)) 
    # nce_losses += nce_loss(x, centroid)
    # nce_losses += nce_loss(y, centroid)
    # nce_losses += nce_loss(z, centroid)

    nce_losses += nce_loss(centroid, x)
    nce_losses += nce_loss(centroid, y)
    nce_losses += nce_loss(centroid, z, mask)
    return nce_losses

# lalign and lunif from https://arxiv.org/pdf/2005.10242.pdf
def lalign(x, y, alpha=2):
    return (x - y).norm(dim=1).pow(alpha).mean()

def lunif(x, t=3):
    sq_pdist = torch.pdist(x, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log()

#implementation from aai/utils/torch/metrics.py
def _compute_mAP(logits, targets, threshold):  
    return torch.masked_select((logits > threshold) == (targets > threshold), (targets > threshold)).float().mean()

#implementation from aai/utils/torch/metrics.py
def compute_mAP(logits, targets, thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)): 
    return torch.mean(torch.stack([_compute_mAP(logits, targets, t) for t in thresholds]))

#implementation from aai/utils/torch/metrics.py
def compute_accuracy(logits, ground_truth, top_k=1):
    """Computes the precision@k for the specified values of k.
    https://github.com/bearpaw/pytorch-classification/blob/cc9106d598ff1fe375cc030873ceacfea0499d77/utils/eval.py
    """
    batch_size = ground_truth.size(0)
    _, pred = logits.topk(top_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(ground_truth.reshape(1, -1).expand_as(pred))
    correct_k = correct[:top_k].reshape(-1).float().sum(0)
    return correct_k.mul_(100.0 / batch_size)

