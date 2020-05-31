from utils.arg_parser import extract_args_from_json
from utils.data_provider import split_dataset, load_movie_categories
from utils.reset_seed import set_seeds
from models import GreedyMLP
from dataloaders.PointwiseDataLoader import PointwiseDataLoader, PointwiseDataLoaderTest
from utils.experiment_builder import ExperimentBuilder

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch


class GreedyMLPExperimentBuilder(ExperimentBuilder):
    criterion = torch.nn.BCELoss()

    def pre_epoch_init_function(self):
        self.train_loader.dataset.negative_sampling()

    def loss_function(self, values):
        """
        Using the loss from the paper https://arxiv.org/pdf/1708.05031.pdf (ie Pointwise loss with negative sampling
        which is binary cross-entropy loss)
        """
        ratings_pred = values[0].double()
        ratings = values[1]

        return self.criterion(ratings_pred.view(-1), ratings)

    def forward_model_training(self, values_to_unpack):
        users = values_to_unpack[0].cuda()
        movies = values_to_unpack[1].cuda()
        ratings = values_to_unpack[2].cuda()

        ratings_pred = self.model(users, movies)

        return self.loss_function((ratings_pred, ratings))

    def forward_model_test(self, values_to_unpack):
        users = values_to_unpack[0].cuda()
        movies = values_to_unpack[1].cuda()

        ratings_pred = self.model(users, movies)
        ratings_pred = ratings_pred.squeeze()

        highest_ratings = torch.topk(ratings_pred, self.configs['slate_size'], dim=1)

        # Return the indices of the movies selected (the slate)
        return highest_ratings[1]


def experiments_run():
    configs = extract_args_from_json()
    set_seeds(configs['seed'])

    df_train, df_val, df_test, df_train_matrix, df_val_matrix, df_test_matrix = split_dataset(configs)
    movies_categories = load_movie_categories(configs)

    train_dataset = PointwiseDataLoader(df_train, df_train_matrix, configs['negative_samples'], movies_categories, True)
    train_loader = DataLoader(train_dataset, batch_size=configs['batch_size'], shuffle=True, num_workers=4,
                              drop_last=True)

    val_dataset = PointwiseDataLoaderTest(df_val, df_val_matrix, configs['negative_samples_per_test_item'],
                                          configs['slate_size'])
    val_loader = DataLoader(val_dataset, batch_size=5, shuffle=True, num_workers=0, drop_last=True)

    model = GreedyMLP.GreedyMLP(len(df_train_matrix.index), len(df_train_matrix.columns), configs['hidden_layers_dims'],
                                configs['use_bias'])

    experiment_builder = GreedyMLPExperimentBuilder(model, train_loader, val_loader, configs)
    experiment_builder.run_experiment()


if __name__ == '__main__':
    experiments_run()
