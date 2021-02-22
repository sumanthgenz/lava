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
    x, y = x.cpu().numpy(), y.cpu().numpy()
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
def nce_loss(x, y):
    sim_matrix = torch.mm(x, y.t())
    pos_pairs = torch.arange(x.size(0)).to(sim_matrix.device)
    loss = torch.nn.functional.cross_entropy(sim_matrix, pos_pairs)
    return loss

def centroid_loss(x, y, z):
    #compute centroid as arithmetic mean of latent vectors x, y, z
    centroid = (x + y + z)/3
    x_loss = l2_distance(x, centroid).mean()
    y_loss = l2_distance(y, centroid).mean()
    z_loss = l2_distance(z, centroid).mean()
    loss = (x_loss + y_loss + z_loss)/3
    return loss

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

if __name__ == "__main__":
    a, v, t = torch.rand(32, 128), torch.rand(32, 128), torch.rand(32, 128)

    print(centroid_loss(a, v, t))
    print(nce_loss(a,v))

    print(*(list(torchvision.models.resnet18(pretrained=True).children())[:-2]))