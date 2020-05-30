import torch.nn as nn
import torch
from torch.optim.adam import Adam
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
import tqdm
import sys

from abc import ABC, abstractmethod


class ExperimentBuilder(nn.Module, ABC):
    def __init__(self, model, train_loader, validation_loader, configs, print_learnable_parameters=True):
        super(ExperimentBuilder, self).__init__()
        self.configs = configs
        self.model = model
        self.model.reset_parameters()

        self.train_loader = train_loader
        self.validation_loader = validation_loader
        # self.test_loader = test_loader

        self.optimizer = Adam(self.parameters(), amsgrad=False, weight_decay=configs['weight_decay'])
        self.device = torch.cuda.current_device()

        if print_learnable_parameters:
            self.print_parameters(self.named_parameters)

        self.set_device(configs['use_gpu'])

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
            self.model.to(self.device)
            self.model = nn.DataParallel(module=self.model)
            print('Use Multi GPU', self.device)
        elif torch.cuda.device_count() == 1 and use_gpu:
            self.device = torch.cuda.current_device()
            self.model.to(self.device)  # sends the model from the cpu to the gpu
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

    def run_validation_epoch(self):
        self.model.eval()

        with torch.no_grad():
            with tqdm.tqdm(total=len(self.validation_loader), file=sys.stdout) as pbar_val:
                for idx, values_to_unpack in enumerate(self.validation_loader):
                    predicted_slate = self.forward_model_test(values_to_unpack)
                    ground_truth_slate = values_to_unpack[2].cuda()

                    print(predicted_slate, ground_truth_slate)

        return np.random.randint(0, 1), np.random.randint(0, 1), np.random.randint(0, 1)

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

            # average_loss = self.run_training_epoch()
            avg_loss, avg_precision, avg_hit_rate = self.run_validation_epoch()

            # val_mean_accuracy = np.mean(current_epoch_losses['val_acc'])

            # if val_mean_accuracy > self.best_val_model_acc:  # if current epoch's mean val acc is greater than the saved best val acc then
            #     self.best_val_model_acc = val_mean_accuracy  # set the best val model acc to be current epoch's val accuracy
            #     self.best_val_model_idx = epoch_idx  # set the experiment-wise best val idx to be the current epoch's idx

            self.writer.add_scalar('Average training loss for epoch', average_loss, epoch_idx)

            # TODO: Validation and testing
            # TODO: Saving model per epoch
