import numpy as np
from torch.utils.data import Dataset


class SlateFormationDataLoader(Dataset):
    def __init__(self, slate_formations, user_movie_matrix):
        self.slate_formations = slate_formations
        self.user_movie_matrix = user_movie_matrix
        self.number_of_movies = len(user_movie_matrix.columns)

        self.user_index = None
        self.slate_vector_matrix = None
        self.response_vector_matrix = None
        self.user_interactions_values = None

        self.convert_to_vector_form()

    def convert_to_vector_form(self):
        self.user_index = np.stack(self.slate_formations['User Index'].values)

        self.slate_vector_matrix = np.stack(self.slate_formations['Slate Movies'].str.split('|').values)
        self.slate_vector_matrix = self.slate_vector_matrix.astype(np.int32)

        self.response_vector_matrix = np.stack(self.slate_formations['Response Vector'].str.split('|').values)
        self.response_vector_matrix = self.response_vector_matrix.astype(np.int32)

        self.user_interactions_values = self.slate_formations['User Interactions'].str.split('|').values

    def __len__(self):
        return len(self.slate_formations)

    def __getitem__(self, idx):
        user_interactions = self.user_interactions_values[idx]
        one_hot = np.zeros((self.number_of_movies,), dtype=int)

        # TODO: If this is slow, create a spare matrix in the init function
        np.put(one_hot, user_interactions, np.ones(len(user_interactions)))

        return self.user_index[idx], one_hot, self.slate_vector_matrix[idx], self.response_vector_matrix[idx]
