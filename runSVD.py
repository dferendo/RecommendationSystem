from utils.arg_parser import extract_args_from_json
from utils.data_provider import split_dataset
from utils.reset_seed import set_seeds
from utils.experiment_builder import ExperimentBuilderNN
from dataloaders.PointwiseDataLoader import PointwiseDataLoader
from dataloaders.TestDataLoader import UserIndexTestDataLoader
import numpy as np

import torch
from torch.utils.data import DataLoader
from surprise import Dataset


class MFExperimentBuilder(ExperimentBuilderNN):
    criterion = torch.nn.MSELoss()

    def pre_epoch_init_function(self):
        # self.train_loader.dataset.negative_sampling()
        pass

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

    train_dataset = PointwiseDataLoader(df_train, df_train_matrix, configs['neg_sample_per_training_example'])
    train_loader = DataLoader(train_dataset, batch_size=configs['train_batch_size'], shuffle=True, num_workers=4,
                              drop_last=True)

    test_dataset = UserIndexTestDataLoader(df_test, df_test_matrix, df_train_matrix)
    test_loader = DataLoader(test_dataset, batch_size=configs['test_batch_size'], shuffle=False, num_workers=4,
                             drop_last=False)

    total_movies = len(df_train_matrix.columns)
    total_users = len(df_train_matrix.index)

    # predicted_slates = []
    # ground_truth_slates = []
    #
    # with tqdm.tqdm(total=len(test_loader), file=sys.stdout) as pbar_val:
    #     for idx, values_to_unpack in enumerate(test_loader):
    #         user_indexes = values_to_unpack[0]
    #
    #         slates = []
    #
    #         for user_index in user_indexes:
    #             user_index = user_index.item()
    #
    #             pred_slates = np.dot(svd_u[user_index], svd_m.T).argsort()[-configs['slate_size']:][::-1]
    #
    #             slates.append(pred_slates)
    #
    #         temp_loop = np.vstack(slates)
    #
    #         ground_truth_slate = values_to_unpack[1].cpu()
    #         ground_truth_indexes = np.nonzero(ground_truth_slate)
    #         grouped_ground_truth = np.split(ground_truth_indexes[:, 1],
    #                                         np.cumsum(np.unique(ground_truth_indexes[:, 0], return_counts=True)[1])[
    #                                         :-1])
    #         predicted_slates.append(temp_loop)
    #         ground_truth_slates.extend(grouped_ground_truth)
    #
    #         pbar_val.update(1)
    #
    # predicted_slates = np.vstack(predicted_slates)
    # predicted_slates = torch.from_numpy(predicted_slates)
    # diversity = movie_diversity(predicted_slates, total_movies)
    #
    # precision, hr = precision_hit_ratio(predicted_slates, ground_truth_slates)
    #
    # print(f'HR: {hr}, Precision: {precision}, Diversity: {diversity}')


if __name__ == '__main__':
    experiments_run()
