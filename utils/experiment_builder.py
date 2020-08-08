import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.evaluation_metrics import precision_hit_coverage_ratio, movie_diversity
from utils.storage import save_statistics

import numpy as np
import os
import tqdm
import sys

from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class ExperimentBuilderNN(nn.Module, ABC):
    def __init__(self, model, train_loader, evaluation_loader, number_of_movies, configs, print_learnable_parameters=True):
        super(ExperimentBuilderNN, self).__init__()
        self.configs = configs
        torch.set_default_tensor_type(torch.FloatTensor)
        self.number_of_movies = number_of_movies

        self.model = model
        self.model.reset_parameters()

        self.train_loader = train_loader
        self.evaluation_loader = evaluation_loader

        self.device = torch.cuda.current_device()
        self.set_device(configs['use_gpu'])

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=configs['lr'], weight_decay=configs['weight_decay'])

        if print_learnable_parameters:
            self.print_parameters(self.named_parameters)

        # Saving runs
        self.experiment_folder = "runs/{0}".format(configs['experiment_name'])

        self.writer = SummaryWriter(self.experiment_folder)
        self.experiment_saved_models = os.path.abspath(os.path.join(self.experiment_folder, "saved_models"))

        if not os.path.exists(self.experiment_saved_models):
            os.mkdir(self.experiment_saved_models)

        self.experiment_logs = os.path.abspath(os.path.join(self.experiment_folder, "result_outputs"))

        if not os.path.exists(self.experiment_logs):
            os.mkdir(self.experiment_logs)

        # Set best models to be at 0 since we are just starting
        self.best_val_model_idx = 0
        self.best_val_model_precision = 0.

        if configs['continue_from_epoch'] != -1:  # if continue from epoch is not -1 then
            self.best_val_model_idx, self.best_val_model_precision, self.state = self.load_model(
                model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                model_idx=configs['continue_from_epoch'])

            self.starting_epoch = self.state['current_epoch_idx']
        else:
            self.starting_epoch = 0
            self.state = dict()

    @staticmethod
    def print_parameters(named_parameters):
        print('System learnable parameters')
        total_num_parameters = 0

        for name, value in named_parameters():
            print(name, value.shape)
            total_num_parameters += np.prod(value.shape)

        print('Total number of parameters', total_num_parameters)

    def set_device(self, use_gpu):
        if torch.cuda.device_count() > 1 and use_gpu:
            self.device = torch.cuda.current_device()

            self.model.to(self.device)
            self.model = nn.DataParallel(module=self.generator)
            print('Use Multi GPU', self.device)
        elif torch.cuda.device_count() == 1 and use_gpu:
            self.device = torch.cuda.current_device()
            self.model.to(self.device)

            print('Use GPU', self.device)
        else:
            print("use CPU")
            self.device = torch.device('cpu')  # sets the device to be CPU
            print(self.device)

    def load_model(self, model_save_dir, model_save_name, model_idx):
        """
        Load the network parameter state and the best val model idx and best val acc to be compared with the future
        val accuracies, in order to choose the best val model
        """
        state = torch.load(f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(model_idx))))
        self.load_state_dict(state_dict=state['network'])

        return state['best_val_model_idx'], state['best_val_model_precision'], state

    def save_model(self, model_save_dir, model_save_name, model_idx, state):
        state['network'] = self.state_dict()
        torch.save(state, f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(model_idx))))

    @abstractmethod
    def pre_epoch_init_function(self):
        """
        Declares or data resets (such as negative sampling) that are done pre-epoch training.
        :return: Nothing
        """
        pass

    @abstractmethod
    def train_iteration(self, idx, values_to_unpack):
        """

        :param idx:
        :param values_to_unpack: Values obtained from the training data loader
        :return: The loss
        """
        pass

    @abstractmethod
    def eval_iteration(self, values_to_unpack):
        """
        :param values_to_unpack: Values obtained from the training data loader
        :return:
        """
        pass

    def run_training_epoch(self):
        self.model.train()
        all_losses = []

        with tqdm.tqdm(total=len(self.train_loader), file=sys.stdout) as pbar:
            for idx, values_to_unpack in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                loss = self.train_iteration(idx, values_to_unpack)

                loss.backward()
                self.optimizer.step()

                all_losses.append(float(loss))

                pbar.update(1)
                pbar.set_description(f"loss: {float(loss):.4f}")

        plt.show()

        return np.mean(all_losses)

    def run_evaluation_epoch(self):
        self.model.eval()
        predicted_slates = []
        ground_truth_slates = []

        with torch.no_grad():
            with tqdm.tqdm(total=len(self.evaluation_loader), file=sys.stdout) as pbar_val:
                for idx, values_to_unpack in enumerate(self.evaluation_loader):
                    predicted_slate = self.eval_iteration(values_to_unpack)

                    ground_truth_slate = values_to_unpack[1].cpu()
                    ground_truth_indexes = np.nonzero(ground_truth_slate)
                    grouped_ground_truth = np.split(ground_truth_indexes[:, 1],
                                                    np.cumsum(np.unique(ground_truth_indexes[:, 0], return_counts=True)[1])[:-1])
                    predicted_slates.append(predicted_slate)
                    ground_truth_slates.extend(grouped_ground_truth)

                    pbar_val.update(1)

        predicted_slates = torch.cat(predicted_slates, dim=0)
        diversity = movie_diversity(predicted_slates, self.number_of_movies)

        predicted_slates = predicted_slates.cpu()
        precision, hr, cc = precision_hit_coverage_ratio(predicted_slates, ground_truth_slates)

        return precision, hr, diversity

    def run_experiment(self):
        total_losses = {"loss": [], "precision": [], "hr": [],
                        "diversity": [], "curr_epoch": []}

        for epoch_idx in range(self.starting_epoch, self.configs['num_of_epochs']):
            print(f"Epoch: {epoch_idx}")
            self.pre_epoch_init_function()

            average_loss = self.run_training_epoch()
            precision_mean, hr_mean, diversity = self.run_evaluation_epoch()

            if precision_mean > self.best_val_model_precision:
                self.best_val_model_precision = precision_mean
                self.best_val_model_idx = epoch_idx

            self.writer.add_scalar('Average training loss per epoch', average_loss, epoch_idx)

            self.writer.add_scalar('Precision', precision_mean, epoch_idx)
            self.writer.add_scalar('Hit Ratio', hr_mean, epoch_idx)
            self.writer.add_scalar('Diversity', diversity, epoch_idx)

            print(f'HR: {hr_mean}, Precision: {precision_mean}, Diversity: {diversity}')

            self.state['current_epoch_idx'] = epoch_idx
            self.state['best_val_model_precision'] = self.best_val_model_precision
            self.state['best_val_model_idx'] = self.best_val_model_idx

            if self.configs['save_model']:
                self.save_model(model_save_dir=self.experiment_saved_models,
                                model_save_name="train_model", model_idx=epoch_idx, state=self.state)

            total_losses['loss'].append(average_loss)
            total_losses['precision'].append(precision_mean)
            total_losses['hr'].append(hr_mean)
            total_losses['diversity'].append(diversity)
            total_losses['curr_epoch'].append(epoch_idx)

            save_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv',
                            stats_dict=total_losses, current_epoch=epoch_idx,
                            continue_from_mode=True if (self.starting_epoch != 0 or epoch_idx > 0) else False)

        self.writer.flush()
        self.writer.close()
