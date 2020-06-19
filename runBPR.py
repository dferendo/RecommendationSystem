from utils.arg_parser import extract_args_from_json
from utils.data_provider import split_dataset
from utils.reset_seed import set_seeds
from dataloaders.TestDataLoader import UserIndexTestDataLoader
from models import BayesianPR
from dataloaders.PairwiseDataLoader import PairwiseDataLoader
from utils.experiment_builder import ExperimentBuilderNN

from torch.utils.data import DataLoader
import torch
import numpy as np


class BPRExperimentBuilder(ExperimentBuilderNN):
    criterion = torch.nn.MSELoss()

    @staticmethod
    def loss_function(predicted_i, predicted_j):
        """
           Using the loss from the paper https://arxiv.org/pdf/1205.2618.pdf
           """
        return ((predicted_i - predicted_j).sigmoid().log().sum()) * -1

    def pre_epoch_init_function(self):
        self.train_loader.dataset.negative_sampling()

    def train_iteration(self, idx, values_to_unpack):
        user_indexes = values_to_unpack[0].to(self.device)
        movie_indexes = values_to_unpack[1].to(self.device)
        negative_movie_indexes = values_to_unpack[2].to(self.device)

        predicted_i, predicted_j = self.model(user_indexes, movie_indexes, negative_movie_indexes)
        loss = self.loss_function(predicted_i, predicted_j)

        return loss

    def eval_iteration(self, values_to_unpack):
        user_indexes = values_to_unpack[0]

        slates = []

        for user_index in user_indexes:
            user_index = user_index.item()

            movie_index = np.arange(self.model.num_items)
            user_index = np.full((self.model.num_items,), user_index)

            movie_tensor = torch.from_numpy(movie_index).to(self.device)
            user_tensor = torch.from_numpy(user_index).to(self.device)

            prediction_i, prediction_j = self.model(user_tensor, movie_tensor, movie_tensor)

            slate = torch.topk(prediction_i, self.configs['slate_size'])

            slates.append(slate.indices)
        predicted_slates = torch.stack(slates, dim=0)

        return predicted_slates


def experiments_run():
    configs = extract_args_from_json()
    print(configs)
    set_seeds(configs['seed'])

    df_train, df_test, df_train_matrix, df_test_matrix, movies_categories = split_dataset(configs)

    train_dataset = PairwiseDataLoader(df_train, df_train_matrix, configs['neg_sample_per_training_example'])

    train_loader = DataLoader(train_dataset, batch_size=configs['train_batch_size'], shuffle=True, num_workers=4,
                              drop_last=False)

    test_dataset = UserIndexTestDataLoader(df_test, df_test_matrix, df_train_matrix)
    test_loader = DataLoader(test_dataset, batch_size=configs['test_batch_size'], shuffle=True, num_workers=4,
                             drop_last=False)

    total_movies = len(df_train_matrix.columns)
    total_users = len(df_train_matrix.index)

    model = BayesianPR.BPR(total_users, total_movies, configs['embed_dims'])

    experiment_builder = BPRExperimentBuilder(model, train_loader, test_loader, total_movies, configs)
    experiment_builder.run_experiment()


if __name__ == '__main__':
    experiments_run()
