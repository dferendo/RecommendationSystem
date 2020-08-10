import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.evaluation_metrics import precision_hit_coverage_ratio, movie_diversity

import numpy as np
import tqdm
import sys

from abc import ABC, abstractmethod
import os


class ExperimentBuilderPlain(nn.Module, ABC):
    def __init__(self, model, evaluation_loader, number_of_movies, movies_categories, titles, configs):
        super(ExperimentBuilderPlain, self).__init__()
        self.configs = configs
        torch.set_default_tensor_type(torch.FloatTensor)
        self.number_of_movies = number_of_movies
        self.movie_categories = movies_categories
        self.titles = titles

        self.model = model
        self.model.reset_parameters()

        self.evaluation_loader = evaluation_loader

        # Saving runs
        self.experiment_folder = "runs/{0}".format(configs['experiment_name'])
        self.writer = SummaryWriter(self.experiment_folder)

        self.predicted_slates = os.path.abspath(os.path.join(self.experiment_folder, "predicted_slate"))

        if not os.path.exists(self.predicted_slates):
            os.mkdir(self.predicted_slates)

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

        path_to_save = os.path.join(self.predicted_slates, f'0.txt')

        with open(path_to_save, 'w') as f:
            for item in predicted_slates:
                f.write(f'{item}\n')

        predicted_slates = predicted_slates.cpu()
        precision, hr, cc = precision_hit_coverage_ratio(predicted_slates, ground_truth_slates, self.movie_categories)

        # Count years
        years_dict = {}
        all_years = np.unique(self.titles)

        for year in all_years:
            years_dict[year] = 0

        for predicted_slate in list(predicted_slates):
            for predicted_movie in predicted_slate:
                years_dict[self.titles[predicted_movie]] += 1

        print(years_dict)

        return precision, hr, cc, diversity

    def run_experiment(self):
        precision_mean, hr_mean, cc, diversity = self.run_evaluation_epoch()

        self.writer.add_scalar('Precision', precision_mean)
        self.writer.add_scalar('Hit Ratio', hr_mean)
        self.writer.add_scalar('Diversity', diversity)
        self.writer.add_scalar('CC', cc)

        print(f'{precision_mean},{hr_mean},{diversity},{cc}')

        self.writer.flush()
        self.writer.close()

