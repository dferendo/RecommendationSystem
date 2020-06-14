import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.evaluation_metrics import precision_hit_ratio, movie_diversity

import numpy as np
import tqdm
import sys

from abc import ABC, abstractmethod


class ExperimentBuilderPlain(nn.Module, ABC):
    def __init__(self, model, evaluation_loader, number_of_movies, configs):
        super(ExperimentBuilderPlain, self).__init__()
        self.configs = configs
        torch.set_default_tensor_type(torch.FloatTensor)
        self.number_of_movies = number_of_movies

        self.model = model
        self.model.reset_parameters()

        self.evaluation_loader = evaluation_loader

        # Saving runs
        self.experiment_folder = "runs/{0}".format(configs['experiment_name'])
        self.writer = SummaryWriter(self.experiment_folder)

    @staticmethod
    def print_parameters(named_parameters):
        print('System learnable parameters')
        total_num_parameters = 0

        for name, value in named_parameters():
            print(name, value.shape)
            total_num_parameters += np.prod(value.shape)

        print('Total number of parameters', total_num_parameters)

    @abstractmethod
    def eval_iteration(self):
        """
        :return:
        """
        pass

    def run_evaluation_epoch(self):
        predicted_slates = []
        ground_truth_slates = []

        with torch.no_grad():
            with tqdm.tqdm(total=len(self.evaluation_loader), file=sys.stdout) as pbar_val:
                for idx, values_to_unpack in enumerate(self.evaluation_loader):
                    predicted_slate = self.eval_iteration()
                    predicted_slates.append(predicted_slate)

                    ground_truth_slate = values_to_unpack
                    # The interactions are returned as an array (size == amount of movies) where the interactions
                    # between user and movie is assigned a 1, otherwise 0.
                    ground_truth_indexes = np.nonzero(ground_truth_slate)

                    # With the above, we have a 2d matrix where the first column is a user (with batch index)
                    # and the second is the movie the user interacted with. We use the following to group by user.
                    grouped_ground_truth = np.split(ground_truth_indexes[:, 1],
                                                    np.cumsum(np.unique(ground_truth_indexes[:, 0], return_counts=True)[1])[:-1])

                    # Extend is needed here since we cannot concat (Ground truth not the same dimensions), thus we will
                    # have a list of tensors with different dimensions
                    ground_truth_slates.extend(grouped_ground_truth)

                    pbar_val.update(1)

        predicted_slates = torch.cat(predicted_slates, dim=0)
        diversity = movie_diversity(predicted_slates, self.number_of_movies)

        predicted_slates = predicted_slates.cpu()
        precision, hr = precision_hit_ratio(predicted_slates, ground_truth_slates)

        return precision, hr, diversity

    def run_experiment(self):
        precision_mean, hr_mean, diversity = self.run_evaluation_epoch()

        self.writer.add_scalar('Precision', precision_mean)
        self.writer.add_scalar('Hit Ratio', hr_mean)
        self.writer.add_scalar('Diversity', diversity)

        print(f'HR: {hr_mean}, Precision: {precision_mean}, Diversity: {diversity}')

        self.writer.flush()
        self.writer.close()

