import numpy as np


class RandomSlateGeneration:
    """
    Generating slates by uniform distribution
    """
    def __init__(self, slate_size, movies_list):
        self.slate_size = slate_size
        self.movies_list = movies_list

    def forward(self):
        return np.random.choice(self.movies_list, self.slate_size)


random_model = RandomSlateGeneration(5, np.arange(500))
print(random_model.forward())