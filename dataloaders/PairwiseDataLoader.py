import numpy as np
from torch.utils.data import Dataset


class PairwiseDataLoader(Dataset):
    def __init__(self, training_examples, train_matrix, neg_sample_per_training_example):
        self.training_examples = training_examples
        self.train_matrix = train_matrix
        self.neg_sample_per_training_example = neg_sample_per_training_example

        self.num_of_movies = len(train_matrix.columns)
        self.all_movies_that_can_be_sampled = np.array(train_matrix.columns)

        self.interactions_with_negative_sampling = None

    def negative_sampling(self):
        assert self.neg_sample_per_training_example > 0

        grouped_users = self.training_examples.groupby(['userId'])['movieId'].apply(list)

        all_negative_samples = []

        for user_id, user_interactions in grouped_users.items():
            # Get the possible index of movieIds that we can sample for this user
            movies_to_sample = np.setxor1d(self.all_movies_that_can_be_sampled, user_interactions)

            # Generate all the negative samples (Not sure about the efficiency of np.choice)
            negative_samples_for_user = np.random.choice(movies_to_sample,
                                                         size=self.neg_sample_per_training_example * len(user_interactions))

            # Reshape so that for every movie, we have x negative samples
            negative_samples_for_user = np.reshape(negative_samples_for_user,
                                                   (len(user_interactions), self.neg_sample_per_training_example))

            users_interactions_matrix = np.expand_dims(np.array(user_interactions), axis=1)
            user_id_matrix = np.full((len(user_interactions), 1), user_id)

            # Concat with the userId, the movieId (positive interaction)
            user_positive_interactions = np.hstack((user_id_matrix, users_interactions_matrix))

            # For every negative sample column, concat it with the user true interaction
            for idx in range(self.neg_sample_per_training_example):
                column = negative_samples_for_user[:, idx]
                column_matrix = np.expand_dims(np.array(column), axis=1)

                # Concat with the user interactions
                negative_samples = np.hstack((user_positive_interactions, column_matrix))

                all_negative_samples.append(negative_samples)

        self.interactions_with_negative_sampling = np.vstack(all_negative_samples)

    def __len__(self):
        return self.neg_sample_per_training_example * len(self.training_examples)

    def __getitem__(self, idx):
        user = self.interactions_with_negative_sampling[idx, 0]
        item_i = self.interactions_with_negative_sampling[idx, 1]
        item_j = self.interactions_with_negative_sampling[idx, 2]

        # Convert from ids to indexes for the embedding
        user = self.train_matrix.index.get_loc(user)
        item_i = self.train_matrix.columns.get_loc(item_i)

        if item_j is not None:
            item_j = self.train_matrix.columns.get_loc(item_j)

        return user, item_i, item_j
