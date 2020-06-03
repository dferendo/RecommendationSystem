import numpy as np
import torch


class RandomSlateGeneration:
    """
    Generating slates by uniform distribution
    """
    def __init__(self, slate_size, movies_indexes, batch_size):
        self.slate_size = slate_size
        self.movies_indexes = movies_indexes
        self.batch_size = batch_size

    def forward(self):
        slates = np.random.choice(self.movies_indexes, size=(self.slate_size * self.batch_size))
        slates = slates.reshape((self.batch_size, self.slate_size))

        return torch.from_numpy(slates)

    def reset_parameters(self):
        pass
