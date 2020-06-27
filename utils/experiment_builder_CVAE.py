import numpy as np
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
import os
from utils.storage import save_statistics
import tqdm
import sys
from utils.evaluation_metrics import precision_hit_ratio, movie_diversity

import torch
import math


def cycle_linear(start, stop, n_epoch, n_cycle, ratio):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):

        v , i = start , 0
        while v <= stop and (int(i+c*period) < n_epoch):
            L[int(i+c*period)] = v
            v += step
            i += 1

    return L


def cycle_sigmoid(start, stop, n_epoch, n_cycle, ratio):
    L = np.ones(n_epoch)
    period = n_epoch / n_cycle
    step = (stop - start) / (period * ratio)  # step is in [0,1]

    # transform into [-6, 6] for plots: v*12.-6.

    for c in range(n_cycle):

        v, i = start, 0
        while v <= stop:
            L[int(i + c * period)] = 1.0 / (1.0 + np.exp(- (v * 12. - 6.)))
            v += step
            i += 1
    return L


def cycle_cosine(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch / n_cycle
    step = (stop - start) / (period * ratio)  # step is in [0,1]

    # transform into [0, pi] for plots:

    for c in range(n_cycle):

        v, i = start, 0
        while v <= stop:
            L[int(i + c * period)] = 0.5 - .5 * math.cos(v * math.pi)
            v += step
            i += 1
    return L


class ExperimentBuilderCVAE(nn.Module):
    def __init__(self, model, train_loader, evaluation_loader, number_of_movies, configs):
        super(ExperimentBuilderCVAE, self).__init__()
        self.configs = configs
        torch.set_default_tensor_type(torch.FloatTensor)
        self.number_of_movies = number_of_movies

        self.model = model
        self.KL_weight = None

        self.train_loader = train_loader
        self.evaluation_loader = evaluation_loader

        self.criterion = torch.nn.CrossEntropyLoss()

        self.device = torch.cuda.current_device()
        self.set_device(configs['use_gpu'])

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=configs['lr'],
                                          weight_decay=configs['weight_decay'])

        # Saving runs
        self.experiment_folder = "runs/{0}".format(configs['experiment_name'])

        self.writer = SummaryWriter(self.experiment_folder)
        self.experiment_saved_models = os.path.abspath(os.path.join(self.experiment_folder, "saved_models"))

        if not os.path.exists(self.experiment_saved_models):
            os.mkdir(self.experiment_saved_models)

        self.experiment_logs = os.path.abspath(os.path.join(self.experiment_folder, "result_outputs"))

        if not os.path.exists(self.experiment_logs):
            os.mkdir(self.experiment_logs)

        self.predicted_slates = os.path.abspath(os.path.join(self.experiment_folder, "predicted_slate"))

        if not os.path.exists(self.predicted_slates):
            os.mkdir(self.predicted_slates)

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

    def loss_function(self, recon_slates, slates, mu, log_variance, prior_mu, prior_log_variance, epoch_idx):
        recon_slates = recon_slates.view(recon_slates.shape[0] * recon_slates.shape[1], recon_slates.shape[2])
        slates = slates.view(slates.shape[0] * slates.shape[1])

        entropy_loss = self.criterion(recon_slates, slates)

        mean_term = ((mu - prior_mu) ** 2) / prior_log_variance.exp()

        KL = 0.5 * torch.sum(prior_log_variance - log_variance + (log_variance.exp() / prior_log_variance.exp()) + mean_term - 1)

        return (KL * self.KL_weight[epoch_idx]) + entropy_loss

    def run_training_epoch(self, epoch_idx):
        self.model.train()
        all_losses = []

        with tqdm.tqdm(total=len(self.train_loader), file=sys.stdout) as pbar:
            for idx, (user_ids, padded_interactions, interactions_length, slates, response_vector) in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                padded_interactions = padded_interactions.to(self.device)
                interactions_length = interactions_length.to(self.device)
                slates = slates.to(self.device).long()
                response_vector = response_vector.to(self.device).float()

                decoder_out, mu, log_variance, prior_mu, prior_log_variance = self.model(slates, padded_interactions,
                                                                                          interactions_length, response_vector)

                loss = self.loss_function(decoder_out, slates, mu, log_variance, prior_mu, prior_log_variance, epoch_idx)

                loss.backward()
                self.optimizer.step()

                all_losses.append(float(loss))

                pbar.update(1)
                pbar.set_description(f"loss: {float(loss):.4f}")

        return np.mean(all_losses)

    def run_evaluation_epoch(self, epoch_idx):
        self.model.eval()
        predicted_slates = []
        ground_truth_slates = []

        with torch.no_grad():
            with tqdm.tqdm(total=len(self.evaluation_loader), file=sys.stdout) as pbar_val:
                for idx, (user_ids, padded_interactions, interaction_length, ground_truth) in enumerate(self.evaluation_loader):
                    padded_interactions = padded_interactions.to(self.device)
                    interaction_length = interaction_length.to(self.device)
                    response_vector = torch.full((padded_interactions.shape[0], self.configs['slate_size']),
                                                 1, device=self.device, dtype=torch.float32)

                    model_slates = self.model.inference(padded_interactions, interaction_length, response_vector)

                    ground_truth_slate = ground_truth.cpu()
                    ground_truth_indexes = np.nonzero(ground_truth_slate)
                    grouped_ground_truth = np.split(ground_truth_indexes[:, 1],
                                                    np.cumsum(np.unique(ground_truth_indexes[:, 0], return_counts=True)[1])[:-1])

                    predicted_slates.append(model_slates)
                    ground_truth_slates.extend(grouped_ground_truth)

                    pbar_val.update(1)

        predicted_slates = torch.cat(predicted_slates, dim=0)
        diversity = movie_diversity(predicted_slates, self.number_of_movies)

        predicted_slates = predicted_slates.cpu()
        precision, hr = precision_hit_ratio(predicted_slates, ground_truth_slates)

        path_to_save = os.path.join(self.predicted_slates, f'{epoch_idx}.txt')

        with open(path_to_save, 'w') as f:
            for item in predicted_slates:
                f.write(f'{item}\n')

        return precision, hr, diversity

    def run_experiment(self):
        total_losses = {"loss": [], "precision": [], "hr": [],
                        "diversity": [], "curr_epoch": []}

        assert self.configs['type'] in ['linear', 'sigmoid', 'cosine']

        if self.configs['type'] == 'linear':
            self.KL_weight = cycle_linear(0.0, self.configs['max_beta'], self.configs['num_of_epochs'], self.configs['cycles'], self.configs['ratio'])
        elif self.configs['type'] == 'sigmoid':
            self.KL_weight = cycle_sigmoid(0.0, self.configs['max_beta'], self.configs['num_of_epochs'], self.configs['cycles'], self.configs['ratio'])
        elif self.configs['type'] == 'cosine':
            self.KL_weight = cycle_cosine(0.0, self.configs['max_beta'], self.configs['num_of_epochs'], self.configs['cycles'], self.configs['ratio'])
        else:
            raise Exception("Give annealing type")

        for epoch_idx in range(self.starting_epoch, self.configs['num_of_epochs']):
            print(f"Epoch: {epoch_idx}")
            average_loss = self.run_training_epoch(epoch_idx)
            precision_mean, hr_mean, diversity = self.run_evaluation_epoch(epoch_idx)

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
