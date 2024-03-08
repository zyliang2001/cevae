import torch
from torch import nn

class CEVAE(nn.Module):
    def __init__(self, x_dim, t_dim, y_dim, hidden_dim, latent_dim):
        return NotImplemented
    
    def construct_encoder(self):
        return NotImplemented
    
    def construct_decoder(self):
        return NotImplemented

    def forward(self, x, t, y):
        return NotImplemented
    
    def prob_loss(self, pred_dist, true_value):
        return NotImplemented

    def prob_loss(self, pred_dist, true_value):
        return NotImplemented
    
    def kl_divergence(self, pred_mu, pred_var, prior_mu=0, prior_var=1):
        return NotImplemented
