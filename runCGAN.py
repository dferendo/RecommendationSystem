from utils.arg_parser import extract_args_from_json
from utils.data_provider import split_dataset, load_movie_categories
from utils.reset_seed import set_seeds
from models import GreedyMLP
from utils.experiment_builder import ExperimentBuilder
from utils.SlateFormation import generate_slate_formation
from dataloaders.SlateFormation import SlateFormationDataLoader
from torch.utils.data import DataLoader
import torch
import os
import pandas as pd


def experiments_run():
    configs = extract_args_from_json()
    set_seeds(configs['seed'])

    # df_train, df_test, df_train_matrix, df_test_matrix = split_dataset(configs)
    # movies_categories = load_movie_categories(configs)

    slate_formation_file_name = 'sf_{}_{}_{}.csv'.format(configs['slate_size'],
                                                         '-'.join(str(e) for e in configs['negative_sampling_for_slates']),
                                                         configs['is_training'])
    slate_formation_file_location = os.path.join(configs['save_location'], slate_formation_file_name)

    # Check if we have the slates for training
    if os.path.isfile(slate_formation_file_location):
        slate_formation = pd.read_csv(slate_formation_file_location)
    else:
        slate_formation = generate_slate_formation(df_train, df_train_matrix, configs['slate_size'],
                                                   configs['negative_sampling_for_slates'],
                                                   slate_formation_file_location)

    train_dataset = SlateFormationDataLoader(slate_formation)
    train_loader = DataLoader(train_dataset, batch_size=configs['batch_size'], shuffle=True, num_workers=4,
                              drop_last=True)

    for idx, values_to_unpack in enumerate(train_loader):
        print(values_to_unpack)


if __name__ == '__main__':
    experiments_run()
