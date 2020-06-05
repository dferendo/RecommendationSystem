import numpy as np
import torch


class PopularKSlateGeneration:
    """
    Generating slates by uniform distribution
    """
    def __init__(self, slate_size, training_interactions, batch_size):
        self.slate_size = slate_size
        self.batch_size = batch_size

        popK = training_interactions.groupby('movieId')['userId']\
            .count()\
            .sort_values(ascending=False)\
            .head(self.slate_size)\
            .index.tolist()

        self.batched_popK = torch.from_numpy(np.tile(popK, (self.batch_size, 1)))

    def forward(self):
        return self.batched_popK
