import torch
import torch.nn as nn


class MF(nn.Module):
    """
    Implementation of MF using the paper https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf
    """
    def __init__(self, num_users, num_items, latent_dims, use_bias=False):
        super(MF, self).__init__()

        self.use_bias = use_bias
        self.embedding_user = torch.nn.Embedding(num_embeddings=num_users, embedding_dim=latent_dims)
        self.embedding_item = torch.nn.Embedding(num_embeddings=num_items, embedding_dim=latent_dims)

        if use_bias:
            self.bias_users = torch.nn.Embedding(num_users, 1)
            self.bias_items = torch.nn.Embedding(num_items, 1)

    def forward(self, user, item):
        out = (self.user_factors(user) * self.item_factors(item))

        if self.use_bias:
            out += self.user_biases(user) + self.item_biases(item)

        return out.sum(1)
