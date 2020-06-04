import numpy as np


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

    def forward(self):
        return np.random.choice(self.movies_list, self.slate_size)

