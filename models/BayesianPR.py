import torch
import torch.nn as nn


class BPR(nn.Module):
    """
    Implementation of BPR-MF (Bayesian Personalized Ranking - Matrix Factorization) following the paper
    https://arxiv.org/pdf/1205.2618.pdf
    """
    def __init__(self, num_users, num_items, latent_dims):
        super(BPR, self).__init__()

        self.embed_user = nn.Embedding(num_users, latent_dims)
        self.embed_item = nn.Embedding(num_items, latent_dims)

    def forward(self, user, item_i, item_j):
        user = self.embed_user(user)
        item_i = self.embed_item(item_i)
        item_j = self.embed_item(item_j)

        # -1 indicates the last dimension
        prediction_i = (user * item_i).sum(dim=-1)
        prediction_j = (user * item_j).sum(dim=-1)

        return prediction_i, prediction_j

    def reset_parameters(self):
        """
        Re-initialize the network parameters.
        """
        self.embed_user.reset_parameters()
        self.embed_item.reset_parameters()
