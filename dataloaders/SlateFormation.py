import numpy as np
from torch.utils.data import Dataset


class SlateFormationDataLoader(Dataset):
    def __init__(self, row_interactions, user_movie_matrix, slate_size, negative_sampling_for_slates,
                 recompute_negative_sample_per_epoch):
        """
        Holds the training examples for slate generation. Each example has a user_id followed by a slate containing
        *slate_size* movie_ids, *slate_size* response vector (whether the user had an interaction or not) and the user
        interactions.
        :param row_interactions: All the interactions between users and movies. Each value contains user_id, movie_id,
        rating and timestamp.
        :param user_movie_matrix: [user_id, movie_id] Sparse DataFrame matrix where user_id are the rows and movie_id
        are the columns. The value of each [user_id, movie_id] is whether there are an interaction.
        :param slate_size: The size of the slate
        :param negative_sampling_for_slates: This is an array where each element indicates how many negative examples
        per slate.
        :param recompute_negative_sample_per_epoch: Whether or not to recompute the slates replacing the negative
        samples for every epoch.
        """
        assert isinstance(negative_sampling_for_slates, list)

        self.row_interactions = row_interactions
        self.user_movie_matrix = user_movie_matrix

        self.slate_size = slate_size
        self.negative_sampling_for_slates = negative_sampling_for_slates
        self.recompute_negative_sample_per_epoch = recompute_negative_sample_per_epoch

        self.all_movies_that_can_be_sampled = np.array(self.user_movie_matrix.columns)

        self.all_slates_interactions = None

        self.negative_sampling()

    def negative_sampling(self):
        if not self.recompute_negative_sample_per_epoch:
            return

        grouped_users = self.row_interactions.groupby(['userId'])['movieId'].apply(list)

        all_samples = []

        for user_id, user_interactions in grouped_users.items():
            # Get the possible index of movieIds that we can sample for this user
            movies_with_no_interactions_with_user = np.setxor1d(self.all_movies_that_can_be_sampled, user_interactions)

            for negative_samples_amount in self.negative_sampling_for_slates:
                assert negative_samples_amount <= self.slate_size

                slate_movies = []
                response_vector = np.zeros(self.slate_size, dtype=np.int32)
                all_user_interactions = self.user_movie_matrix.loc[user_id].to_list()

                positive_samples_amount = self.slate_size - negative_samples_amount

                if positive_samples_amount != 0:
                    positive_samples = user_interactions[-positive_samples_amount:]

                    response_vector[:positive_samples_amount] = 1

                    # Convert to indices
                    positive_indexes = list(map(lambda movie_id: self.user_movie_matrix.columns.get_loc(movie_id),
                                                positive_samples))

                    slate_movies.extend(positive_indexes)

                    # Remove positive samples that are in the slate from user interactions
                    for positive_index in positive_indexes:
                        all_user_interactions[positive_index] = 0

                if negative_samples_amount != 0:
                    negative_samples = np.random.choice(movies_with_no_interactions_with_user,
                                                        size=negative_samples_amount)

                    # Convert to indices
                    negative_indexes = list(map(lambda movie_id: self.user_movie_matrix.columns.get_loc(movie_id),
                                                negative_samples))

                    slate_movies.extend(negative_indexes)

                user_index = self.user_movie_matrix.index.get_loc(user_id)

                sample = [user_index]

                sample.extend(all_user_interactions)
                sample.extend(slate_movies)
                sample.extend(response_vector)

                print(len(sample))

                all_samples.append([user_index, all_user_interactions, slate_movies, response_vector])

        print(all_samples[0])

    def __len__(self):
        if self.is_training:
            return self.neg_sample_per_training_example * len(self.training_examples)

        return len(self.training_examples)

    def __getitem__(self, idx):
        # If in training, we need to use the negatively sampled item with the interacted item
        if self.is_training:
            user = self.all_interactions[idx, 0]
            item_i = self.all_interactions[idx, 1]
        else:
            user = self.training_examples.iloc[idx]['userId']
            item_i = self.training_examples.iloc[idx]['movieId']

        # Convert from ids to indexes for the embedding
        user_index = self.train_matrix.index.get_loc(user)
        item_i_index = self.train_matrix.columns.get_loc(item_i)

        # Obtain the rating from the original matrix since samples can be positive/negative (both user/item are
        # indicating indices, thus you need to use iat to get the rating)
        rating = self.train_matrix.iat[user_index, item_i_index]

        return user_index, item_i_index, rating
