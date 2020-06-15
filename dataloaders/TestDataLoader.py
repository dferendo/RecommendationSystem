from torch.utils.data import Dataset


class NoAdditionalInfoTestDataLoader(Dataset):
    def __init__(self, row_interactions, user_movie_matrix):
        self.row_interactions = row_interactions
        self.user_movie_matrix = user_movie_matrix.to_numpy()

    def __len__(self):
        return self.user_movie_matrix.shape[0]

    def __getitem__(self, idx):
        print(idx)
        return self.user_movie_matrix[idx]


class UserIndexTestDataLoader(Dataset):
    def __init__(self, row_interactions, user_movie_matrix):
        self.row_interactions = row_interactions
        self.user_movie_matrix = user_movie_matrix.to_numpy()

    def __len__(self):
        return self.user_movie_matrix.shape[0]

    def __getitem__(self, idx):
        return idx, self.user_movie_matrix[idx]
