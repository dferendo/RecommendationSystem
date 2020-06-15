import torch
import torch.nn as nn


class MF(nn.Module):
    """
    Implementation of MF using the paper https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf
    """
    def __init__(self, num_users, num_items, latent_dims, use_bias=False):
        super(MF, self).__init__()

        self.num_users = num_users
        self.num_items = num_items

        self.use_bias = use_bias
        self.embedding_user = torch.nn.Embedding(num_embeddings=num_users, embedding_dim=latent_dims)
        self.embedding_item = torch.nn.Embedding(num_embeddings=num_items, embedding_dim=latent_dims)

        if use_bias:
            self.bias_users = torch.nn.Embedding(num_users, 1)
            self.bias_items = torch.nn.Embedding(num_items, 1)

    def forward(self, user, item):
        out = (self.embedding_user(user) * self.embedding_item(item))

        if self.use_bias:
            out += self.bias_users(user) + self.bias_items(item)

        return out.sum(dim=1)

    def reset_parameters(self):
        """
        Re-initializes the networks parameters
        """
        self.embedding_user.reset_parameters()
        self.embedding_item.reset_parameters()
