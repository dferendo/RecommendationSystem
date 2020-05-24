from utils.arg_parser import extract_args_from_json
from utils.data_provider import split_dataset
from utils.reset_seed import set_seeds
from models import PopK

import pandas as pd


def experiments_run():
    configs = extract_args_from_json()
    set_seeds(configs['seed'])

    df_train, df_val, df_test = split_dataset(configs)
    df_all_training = pd.concat([df_train, df_val])
    unique_movies = df_all_training['movieId'].unique()

    popK_model = PopK.PopularKSlateGeneration(configs['slate_size'], unique_movies)

    # TODO: evaluation


if __name__ == '__main__':
    experiments_run()
