from utils.arg_parser import extract_args_from_json
from utils.data_provider import split_dataset
from utils.reset_seed import set_seeds
from models import BayesianPR

import numpy as np

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.optim.adam import Adam


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

    df_train, df_val, df_test, df_train_matrix = split_dataset(configs)

    train_dataset = BPRData(df_train, df_train_matrix, configs['negative_samples'], configs['use_bias'])
    train_loader = DataLoader(train_dataset, batch_size=configs['batch_size'], shuffle=True, num_workers=4)

    model = BayesianPR.BPR(len(df_train_matrix.index), len(df_train_matrix.columns), configs['hidden_dims'])
    model.reset_parameters()

    if torch.cuda.device_count() > 1 and configs['use_gpu']:
        device = torch.cuda.current_device()
        model.to(device)
        model = nn.DataParallel(module=model)
        print('Use Multi GPU', device)
    elif torch.cuda.device_count() == 1 and configs['use_gpu']:
        device = torch.cuda.current_device()
        model.to(device)  # sends the model from the cpu to the gpu
        print('Use GPU', device)
    else:
        print("use CPU")
        device = torch.device('cpu')  # sets the device to be CPU
        print(device)

    model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.lamda)
    self.optimizer = Adam(self.parameters(), amsgrad=False, weight_decay=weight_decay_coefficient)

    # TODO: evaluation


if __name__ == '__main__':
    experiments_run()
