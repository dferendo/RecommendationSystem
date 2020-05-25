import torch
import torch.nn as nn


class BPR(nn.Module):
    """
    Implementation of BPR-MF (Bayesian Personalized Ranking - Matrix Factorization) following the paper
    https://arxiv.org/pdf/1205.2618.pdf
    """
    def __init__(self):
        super(BPR, self).__init__()

        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

    def forward(self, user, item_i, item_j):
        user = self.embed_user(user)
        item_i = self.embed_item(item_i)
        item_j = self.embed_item(item_j)

        prediction_i = (user * item_i).sum(dim=-1)
        prediction_j = (user * item_j).sum(dim=-1)

        return prediction_i, prediction_j