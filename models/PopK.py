import numpy as np
import torch


class PopularKSlateGeneration:
    """
    Generating slates by uniform distribution
    """
    def __init__(self, slate_size, training_interactions, user_movie_matrix, batch_size):
        self.slate_size = slate_size
        self.batch_size = batch_size
        self.user_movie_matrix = user_movie_matrix

        most_popular_movie_ids = training_interactions.groupby('movieId')['userId']\
            .count()\
            .sort_values(ascending=False)\
            .head(self.slate_size)\
            .index.tolist()

        most_popular_movie_index = list(map(lambda movie_id: self.user_movie_matrix.columns.get_loc(movie_id),
                                            most_popular_movie_ids))

        self.batched_popK = torch.from_numpy(np.tile(most_popular_movie_index, (self.batch_size, 1)))

    def forward(self):
        return self.batched_popK

    def reset_parameters(self):
        pass
