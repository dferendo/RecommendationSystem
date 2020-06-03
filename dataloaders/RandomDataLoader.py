from torch.utils.data import Dataset
import numpy as np


class NoAdditionalInfoTestDataLoader(Dataset):
    def __init__(self, row_interactions, user_movie_matrix):
        self.row_interactions = row_interactions
        self.user_movie_matrix = user_movie_matrix

    def __len__(self):
        return len(self.user_movie_matrix.index)

    def __getitem__(self, idx):
        return np.array(self.user_movie_matrix.iloc[idx])
