from utils.arg_parser import extract_args_from_json
from utils.data_provider import split_dataset
from utils.reset_seed import set_seeds
from utils.experiment_builder import ExperimentBuilderNN
from dataloaders.TestDataLoader import UserIndexTestDataLoader
from utils.evaluation_metrics import precision_hit_coverage_ratio, movie_diversity
import numpy as np

import torch
from torch.utils.data import DataLoader
import implicit
from scipy import sparse


class MFExperimentBuilder(ExperimentBuilderNN):
    criterion = torch.nn.MSELoss()

    def pre_epoch_init_function(self):
        self.train_loader.dataset.negative_sampling()

    def train_iteration(self, idx, values_to_unpack):
        user_indexes = values_to_unpack[0].to(self.device)
        movie_indexes = values_to_unpack[1].to(self.device)
        ratings = values_to_unpack[2].to(self.device).float()

        predicted = self.model(user_indexes, movie_indexes)
        loss = self.criterion(predicted, ratings)

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

            prediction = self.model(user_tensor, movie_tensor)

            slate = torch.topk(prediction, self.configs['slate_size'])

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

    model = implicit.als.AlternatingLeastSquares(regularization=configs['weight_decay'], iterations=50,
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

        precision, hr, cc = precision_hit_coverage_ratio(predicted_slates, ground_truth_slates, movies_categories)
        diversity = movie_diversity(predicted_slates, len(df_train_matrix.columns))

        print(precision, hr, cc)
        print(diversity)


if __name__ == '__main__':
    experiments_run()
