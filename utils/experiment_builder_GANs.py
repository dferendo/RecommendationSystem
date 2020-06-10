import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.evaluation_metrics import hit_ratio, precision, category_coverage

import numpy as np
import os
import tqdm
import sys

from abc import ABC, abstractmethod


class ExperimentBuilderGAN(nn.Module, ABC):
    def __init__(self, generator, discriminator, train_loader, evaluation_loader, configs,
                 print_learnable_parameters=True):
        super(ExperimentBuilderGAN, self).__init__()
        self.configs = configs

        self.generator = generator
        self.discriminator = discriminator

        self.generator.reset_parameters()
        self.discriminator.reset_parameters()

        self.train_loader = train_loader
        self.evaluation_loader = evaluation_loader

        self.device = torch.cuda.current_device()
        self.set_device(configs['use_gpu'])

        self.optimizer_gen = torch.optim.Adam(self.generator.parameters(), lr=configs['learning_rate_gen'])
        self.optimizer_dis = torch.optim.Adam(self.discriminator.parameters(), lr=configs['learning_rate_dis'])

        if print_learnable_parameters:
            self.print_parameters(self.named_parameters)

        # Saving runs
        self.experiment_folder = "runs/{0}".format(configs['experiment_name'])

        self.writer = SummaryWriter(self.experiment_folder)
        self.experiment_saved_models = os.path.abspath(os.path.join(self.experiment_folder, "saved_models"))

        if not os.path.exists(self.experiment_saved_models):
            os.mkdir(self.experiment_saved_models)

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

            self.generator.to(self.device)
            self.discriminator.to(self.device)

            self.generator = nn.DataParallel(module=self.generator)
            self.discriminator = nn.DataParallel(module=self.discriminator)
            print('Use Multi GPU', self.device)
        elif torch.cuda.device_count() == 1 and use_gpu:
            self.device = torch.cuda.current_device()

            self.generator.to(self.device)
            self.discriminator.to(self.device)

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
    def loss_function(self, values):
        """
        :param values: Represents a tuple of values
        :return: Loss
        """
        pass

    @abstractmethod
    def forward_model_training(self, values_to_unpack):
        """

        :param values_to_unpack: Values obtained from the training data loader
        :return: The loss
        """
        pass

    @abstractmethod
    def forward_model_test(self, values_to_unpack):
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
                self.model.zero_grad()
                loss = self.forward_model_training(values_to_unpack)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_value = loss.data.detach().cpu().numpy()

                all_losses.append(float(loss_value))
                pbar.update(1)
                pbar.set_description(f"loss: {float(loss_value):.4f}")

        return np.mean(all_losses)

    def run_evaluation_epoch(self, evaluation_loader):
        self.model.eval()
        predicted_slates = []
        ground_truth_slates = []

        with torch.no_grad():
            with tqdm.tqdm(total=len(evaluation_loader), file=sys.stdout) as pbar_val:
                for idx, values_to_unpack in enumerate(evaluation_loader):
                    predicted_slate = self.forward_model_test(values_to_unpack)
                    ground_truth_slate = values_to_unpack[2].cuda()

                    predicted_slates.append(predicted_slate)
                    ground_truth_slates.append(ground_truth_slate)

                    pbar_val.update(1)

                predicted_slates = torch.cat(predicted_slates, dim=0).cpu()
                ground_truth_slates = torch.cat(ground_truth_slates, dim=0).cpu()

        return hit_ratio(predicted_slates, ground_truth_slates), precision(predicted_slates, ground_truth_slates), \
               category_coverage(predicted_slates, self.train_loader)

    def run_experiment(self):
        # Save hyper-parameters
        with SummaryWriter() as w:
            hyper_params = self.configs.copy()

            # An error will be thrown if a value of a param is an array
            for key, value in hyper_params.items():
                if type(value) is list:
                    hyper_params[key] = ''.join(map(str, value))

            w.add_hparams(hyper_params, {})

        for epoch_idx in range(self.starting_epoch, self.configs['num_of_epochs']):
            self.pre_epoch_init_function()

            average_loss = self.run_training_epoch()
            hr_mean, precision_mean, cat_cov_mean = self.run_evaluation_epoch(self.validation_loader)

            if precision_mean > self.best_val_model_precision:
                self.best_val_model_precision = precision_mean
                self.best_val_model_idx = epoch_idx

            self.writer.add_scalar('Average training loss for epoch', average_loss, epoch_idx)
            self.writer.add_scalar('Hit Ratio', hr_mean, epoch_idx)
            self.writer.add_scalar('Precision', precision_mean, epoch_idx)
            self.writer.add_scalar('Category Coverage', cat_cov_mean, epoch_idx)

            print(f'HR: {hr_mean}, Precision: {precision_mean}, Category Coverage: {cat_cov_mean}')

            self.state['current_epoch_idx'] = epoch_idx
            self.state['best_val_model_precision'] = self.best_val_model_precision
            self.state['best_val_model_idx'] = self.best_val_model_idx

            self.save_model(model_save_dir=self.experiment_saved_models,
                            model_save_name="train_model", model_idx=epoch_idx, state=self.state)
            self.save_model(model_save_dir=self.experiment_saved_models,
                            model_save_name="train_model", model_idx='latest', state=self.state)

        print(f"Generating test set evaluation metrics, best model on validation was Epoch {self.best_val_model_idx}")

        self.load_model(model_save_dir=self.experiment_saved_models, model_idx=self.best_val_model_idx,
                        model_save_name="train_model")

        hr_mean, precision_mean, cat_cov_mean = self.run_evaluation_epoch(self.test_loader)

        self.writer.add_scalar('Test: Hit Ratio', hr_mean, 0)
        self.writer.add_scalar('Test: Precision', precision_mean, 0)
        self.writer.add_scalar('Test: Category Coverage', cat_cov_mean, 0)

        print(f'HR: {hr_mean}, Precision: {precision_mean}, Category Coverage: {cat_cov_mean}')

        self.writer.flush()
        self.writer.close()
