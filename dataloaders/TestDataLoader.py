from torch.utils.data import Dataset


class NoAdditionalInfoTestDataLoader(Dataset):
    def __init__(self, row_interactions, user_movie_matrix):
        self.row_interactions = row_interactions
        self.user_movie_matrix = user_movie_matrix.to_numpy()

    def __len__(self):
        return self.user_movie_matrix.shape[0]

    def __getitem__(self, idx):
        return self.user_movie_matrix[idx]


class UserIndexTestDataLoader(Dataset):
    def __init__(self, row_interactions, user_movie_matrix, train_user_movie_matrix):
        self.row_interactions = row_interactions
        self.user_ids = user_movie_matrix.index.to_list()

        self.user_movie_matrix = user_movie_matrix.to_numpy()
        self.train_user_movie_matrix = train_user_movie_matrix

    def __len__(self):
        return self.user_movie_matrix.shape[0]

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        user_index_in_train = self.train_user_movie_matrix.index.get_loc(user_id)

        return user_index_in_train, self.user_movie_matrix[idx]
