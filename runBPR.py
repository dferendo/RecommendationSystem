from utils.arg_parser import extract_args_from_json
from utils.data_provider import split_dataset
from utils.reset_seed import set_seeds
from dataloaders.TestDataLoader import UserIndexTestDataLoader
from models import BayesianPR
from dataloaders.PairwiseDataLoader import PairwiseDataLoader
from utils.experiment_builder import ExperimentBuilderNN
from utils.evaluation_metrics import precision_hit_ratio, movie_diversity

from torch.utils.data import DataLoader
import torch
import numpy as np
import implicit
from scipy import sparse


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

    test_dataset = UserIndexTestDataLoader(df_test, df_test_matrix, df_train_matrix)
    test_loader = DataLoader(test_dataset, batch_size=configs['test_batch_size'], shuffle=True, num_workers=4,
                             drop_last=False)

    model = implicit.bpr.BayesianPersonalizedRanking(learning_rate=configs['lr'], regularization=configs['weight_decay'], iterations=50,
                                                     factors=configs['embed_dims'])

    a = sparse.coo_matrix(df_train_matrix.to_numpy().T)
    temp = sparse.csr_matrix(df_train_matrix.to_numpy())

    # train the model on a sparse matrix of item/user/confidence weights
    model.fit(a)

    for slate_size in configs['slate_size']:
        print(f'Test for {slate_size}')
        recommendations = model.recommend_all(temp, N=slate_size)

        predicted_slates = []
        ground_truth_slates = []

        for values in test_loader:
            for value in values[0]:
                predicted_slates.append(recommendations[int(value)])

            ground_truth_slate = values[1].cpu()
            ground_truth_indexes = np.nonzero(ground_truth_slate)
            grouped_ground_truth = np.split(ground_truth_indexes[:, 1],
                                            np.cumsum(np.unique(ground_truth_indexes[:, 0], return_counts=True)[1])[:-1])

            ground_truth_slates.extend(grouped_ground_truth)

        predicted_slates = torch.from_numpy(np.vstack(predicted_slates))

        precision, hr = precision_hit_ratio(predicted_slates, ground_truth_slates)
        diversity = movie_diversity(predicted_slates, len(df_train_matrix.columns))

        print(precision, hr)
        print(diversity)


if __name__ == '__main__':
    experiments_run()
