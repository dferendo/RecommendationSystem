from utils.arg_parser import extract_args_from_json
from utils.data_provider import split_dataset
from utils.reset_seed import set_seeds
from models import Random

import pandas as pd
import numpy as np


def experiments_run():
    configs = extract_args_from_json()
    set_seeds(configs['seed'])

    df_train, df_val, df_test = split_dataset(configs)

    R_df = df_train.pivot(index='userId', columns='movieId', values='rating').fillna(0)

    R = R_df.to_numpy()
    # Why do we need to mean the data?
    user_ratings_mean = np.mean(R, axis=1)
    R_demeaned = R - user_ratings_mean.reshape(-1, 1)

    print(R_demeaned)
    # df_all_training = pd.concat([df_train, df_val])
    # unique_movies = df_all_training['movieId'].unique()
    #
    # random_model = Random.RandomSlateGeneration(configs['slate_size'], unique_movies)
    #
    # TODO: evaluation


if __name__ == '__main__':
    experiments_run()

# MF()
# adagrad_loss = torch.optim.Adagrad(model.parameters(), lr=1e-6)
#
#
# model = MatrixFactorization(n_users, n_items, n_factors=20)
# loss_fn = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)