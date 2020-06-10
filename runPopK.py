from utils.arg_parser import extract_args_from_json
from utils.data_provider import split_dataset
from utils.reset_seed import set_seeds
from dataloaders.TestDataLoader import NoAdditionalInfoTestDataLoader
from models.PopK import PopularKSlateGeneration
from utils.evaluation_metrics import hit_ratio, precision, category_coverage

from torch.utils.data import DataLoader
import tqdm
import numpy as np
import sys
import torch


def experiments_run():
    configs = extract_args_from_json()
    set_seeds(configs['seed'])

    df_train, df_test, df_train_matrix, df_test_matrix, movies_categories = split_dataset(configs)

    test_dataset = NoAdditionalInfoTestDataLoader(df_test, df_test_matrix)
    test_loader = DataLoader(test_dataset, batch_size=configs['test_batch_size'],
                             shuffle=True, num_workers=4, drop_last=True)

    model = PopularKSlateGeneration(configs['slate_size'], df_train, configs['test_batch_size'])

    predicted_slates = []
    ground_truth_slates = []

    with tqdm.tqdm(total=len(test_loader), file=sys.stdout) as pbar:
        for idx, ground_truth_interactions in enumerate(test_loader):
            # The slate contains the indexes of the movies selected (Not movie Ids)
            predicted_slate = model.forward()

            predicted_slates.append(predicted_slate)

            # The interactions are returned as an array (size == amount of movies) where the interactions between
            # user and movie is assigned a 1, otherwise 0.
            ground_truth_indexes = np.nonzero(ground_truth_interactions)

            # With the above, we have a 2d matrix where the first column is a user (with batch index) and the second is
            # the movie the user interacted with. We use the following to group by user.
            grouped_ground_truth = np.split(ground_truth_indexes[:, 1],
                                            np.cumsum(np.unique(ground_truth_indexes[:, 0], return_counts=True)[1])[:-1])

            # Extend is needed here since we cannot concat (Ground truth not the same dimensions), thus we will
            # have a list of tensors with different dimensions
            ground_truth_slates.extend(grouped_ground_truth)

            pbar.update(1)

    predicted_slates = torch.cat(predicted_slates, dim=0)

    print("Recall: ", hit_ratio(predicted_slates, grouped_ground_truth))
    print("Precision: ", precision(predicted_slates, grouped_ground_truth))
    print("Category Coverage: ", category_coverage(predicted_slates, movies_categories))


if __name__ == '__main__':
    experiments_run()
