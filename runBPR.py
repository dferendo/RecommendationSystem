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
        self.all_movies_that_can_be_sampled = np.array(train_matrix.columns)

        self.interactions_with_negative_sampling = None

    def negative_sampling(self):
        # Sampling is only needed when training
        assert self.is_training
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

    train_dataset = BPRData(df_train, df_train_matrix, configs['negative_samples'], True)
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

    optimizer = Adam(model.parameters(), amsgrad=False, weight_decay=1e-05)

    for epoch in range(10):
        model.train()
        # Get negative sampling
        train_loader.dataset.negative_sampling()

        for user, item_i, item_j in train_loader:
            print(user, item_i, item_j)

        print(epoch)

    # TODO: evaluation


if __name__ == '__main__':
    experiments_run()
