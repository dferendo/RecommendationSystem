import numpy as np
from torch.utils.data import Dataset


class PointwiseDataLoader(Dataset):
    """
    For pointwise sampling, we follow the paper https://arxiv.org/pdf/1708.05031.pdf where we will uniformly sample
    negative samples using unobserved interactions using a ratio.
    """
    def __init__(self, training_examples, train_matrix, neg_sample_per_training_example):
        self.training_examples = training_examples
        self.train_matrix = train_matrix
        self.neg_sample_per_training_example = neg_sample_per_training_example

        self.all_movies_that_can_be_sampled = np.array(train_matrix.columns)

        # This will contain a matrix [user_id, item_id]. Note the item_id can be a positive or negative interaction
        self.all_interactions = None

        self.negative_sampling()

    def negative_sampling(self):
        grouped_users = self.training_examples.groupby(['userId'])['movieId'].apply(list)
        all_samples = []

        for user_id, user_interactions in grouped_users.items():
            # Get the possible index of movieIds that we can sample for this user
            movies_to_sample = np.setxor1d(self.all_movies_that_can_be_sampled, user_interactions)

            users_interactions_matrix = np.expand_dims(np.array(user_interactions), axis=1)
            user_id_matrix = np.full((len(user_interactions), 1), user_id)

            # Concat with the userId, the movieId (positive interaction)
            user_positive_interactions = np.hstack((user_id_matrix, users_interactions_matrix))

            all_samples.append(user_positive_interactions)

            if self.neg_sample_per_training_example == 0:
                continue

            # Generate all the negative samples (Not sure about the efficiency of np.choice)
            negative_samples_for_user = np.random.choice(movies_to_sample,
                                                         size=self.neg_sample_per_training_example * len(user_interactions))

            # Reshape so that for every interaction, we have x negative samples
            negative_samples_for_user = np.reshape(negative_samples_for_user,
                                                   (len(user_interactions), self.neg_sample_per_training_example))

            # For every negative sample column, concat it with the user true interaction
            for idx in range(self.neg_sample_per_training_example):
                column = negative_samples_for_user[:, idx]
                column_matrix = np.expand_dims(np.array(column), axis=1)

                # Concat with the user interactions
                negative_samples = np.hstack((user_id_matrix, column_matrix))

                all_samples.append(negative_samples)

        self.all_interactions = np.vstack(all_samples)

    def __len__(self):
        if self.neg_sample_per_training_example == 0:
            return len(self.training_examples)

        return self.neg_sample_per_training_example * len(self.training_examples)

    def __getitem__(self, idx):
        user = self.all_interactions[idx, 0]
        item_i = self.all_interactions[idx, 1]

        # Convert from ids to indexes for the embedding
        user_index = self.train_matrix.index.get_loc(user)
        item_i_index = self.train_matrix.columns.get_loc(item_i)

        # Obtain the rating from the original matrix since samples can be positive/negative (both user/item are
        # indicating indices, thus you need to use iat to get the rating)
        rating = self.train_matrix.iat[user_index, item_i_index]

        return user_index, item_i_index, rating
