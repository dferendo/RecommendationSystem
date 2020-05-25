from utils.arg_parser import extract_args_from_json
from utils.data_provider import split_dataset
from utils.reset_seed import set_seeds
from models import Random

import pandas as pd
import numpy as np

from torch.utils.data import Dataset


class BPRData(Dataset):
    def __init__(self, training_examples, train_matrix, neg_sample_per_training_example, is_training):
        self.training_examples = training_examples
        self.train_matrix = train_matrix
        self.neg_sample_per_training_example = neg_sample_per_training_example
        self.is_training = is_training
        self.num_of_movies = len(train_matrix.columns)

        self.features_fill = []

    def negative_sampling(self):
        # Sampling is only needed when training
        assert self.is_training

        for index, interaction in self.training_examples.iterrows():
            user, movie = interaction['userId'], interaction['movieId']
            user_movies = self.train_matrix.loc[user]

            for idx in range(self.neg_sample_per_training_example):
                negative_sampled_movie = np.random.randint(self.num_of_movies)

                # In this case, we need to use iloc since the random number generated that not correspond to a movieId
                # but rather to an array index
                while user_movies.iloc[negative_sampled_movie] != 0:
                    negative_sampled_movie = np.random.randint(self.num_of_movies)

                self.features_fill.append([user, movie, user_movies.index[negative_sampled_movie]])

    def __len__(self):
        if self.is_training:
            return self.neg_sample_per_training_example * len(self.training_examples)

        return len(self.training_examples)

    def __getitem__(self, idx):
        # If in training, we need to use the negatively sampled item with the interacted item
        if self.is_training:
            features = self.features_fill
        else:
            features = self.training_examples

        # TODO:
        user = features[idx][0]
        item_i = features[idx][1]
        item_j = features[idx][2] if \
            self.is_training else features[idx][1]
        return user, item_i, item_j


def experiments_run():
    configs = extract_args_from_json()
    set_seeds(configs['seed'])

    df_train, df_val, df_test, df_train_sparse = split_dataset(configs)

    BPRData(df_train, df_train_sparse, 5, True).negative_sampling()

    # R_df = df_train.pivot(index='userId', columns='movieId', values='rating').fillna(0)

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     print(R_df.columns)

    # R = R_df.to_numpy()
    # # Why do we need to mean the data?
    # user_ratings_mean = np.mean(R, axis=1)
    # R_demeaned = R - user_ratings_mean.reshape(-1, 1)
    #
    # print(R_demeaned)
    # df_all_training = pd.concat([df_train, df_val])
    # unique_movies = df_all_training['movieId'].unique()
    #
    # random_model = Random.RandomSlateGeneration(configs['slate_size'], unique_movies)
    #
    # TODO: evaluation


if __name__ == '__main__':
    experiments_run()
