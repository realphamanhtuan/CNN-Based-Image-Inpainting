import torch

def loss_l1(pos, neg):
    hinge_pos = torch.mean(torch.relu(1-pos))
    hinge_neg = torch.mean(torch.relu(1+neg))
    return 0.5 * hinge_pos + 0.5 * hinge_neg   

def loss_sngan(neg):
    g_loss = -torch.mean(neg)
    return g_loss