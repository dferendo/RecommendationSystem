import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
import os
from utils.storage import save_statistics
import tqdm
import sys
import numpy as np
from utils.evaluation_metrics import precision_hit_ratio, movie_diversity
import matplotlib.pyplot as plt
from torch.autograd import Variable
from graphviz import Digraph

from graphviz import Digraph
import torch
from torch.autograd import Variable


def make_dot(var, params):
    """ Produces Graphviz representation of PyTorch autograd graph.

    Blue nodes are trainable Variables (weights, bias).
    Orange node are saved tensors for the backward pass.

    Args:
        var: output Variable
        params: list of (name, Parameters)
    """

    param_map = {id(v): k for k, v in params}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')

    dot = Digraph(
        filename='network',
        format='pdf',
        node_attr=node_attr,
        graph_attr=dict(size="12,12"))
    seen = set()

    def add_nodes(var):
        if var not in seen:

            node_id = str(id(var))

            if torch.is_tensor(var):
                node_label = "saved tensor\n{}".format(tuple(var.size()))
                dot.node(node_id, node_label, fillcolor='orange')

            elif hasattr(var, 'variable'):
                variable_name = param_map.get(id(var.variable))
                variable_size = tuple(var.variable.size())
                node_name = "{}\n{}".format(variable_name, variable_size)
                dot.node(node_id, node_name, fillcolor='lightblue')

            else:
                node_label = type(var).__name__.replace('Backward', '')
                dot.node(node_id, node_label)

            seen.add(var)

            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])

            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(var.grad_fn)

    return dot


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)


class ExperimentBuilderCVAE(nn.Module):
    def __init__(self, model, train_loader, evaluation_loader, number_of_movies, configs):
        super(ExperimentBuilderCVAE, self).__init__()
        self.configs = configs
        torch.set_default_tensor_type(torch.FloatTensor)
        self.number_of_movies = number_of_movies

        self.model = model

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

    def loss_function(self, recon_slates, slates, mu, log_variance, prior_mu, prior_log_variance):
        recon_slates = recon_slates.view(recon_slates.shape[0] * recon_slates.shape[1], recon_slates.shape[2])
        slates = slates.view(slates.shape[0] * slates.shape[1])

        entropy_loss = self.criterion(recon_slates, slates)

        def kl_divergence(Q_mean, Q_log_var, P_mean, P_log_var):
            # noinspection PyTypeChecker
            P_var_inverse = 1 / torch.exp(P_log_var)
            var_ratio_term = torch.exp(Q_log_var) * P_var_inverse

            N_mean = Q_mean - P_mean
            mean_term = N_mean.pow(2) * P_var_inverse

            kl = 0.5 * torch.sum(P_log_var - 1 - Q_log_var + var_ratio_term + mean_term)

            return kl

        KL = kl_divergence(mu, log_variance, prior_mu, prior_log_variance)

        print(entropy_loss)

        return (KL * self.configs['beta_weight']) + entropy_loss

    def run_training_epoch(self):
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


                loss = self.loss_function(decoder_out, slates, mu, log_variance, prior_mu, prior_log_variance)

                loss.backward()
                plot_grad_flow(self.model.named_parameters())
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

        predicted_slates = torch.cat(predicted_slates, dim=0)
        diversity = movie_diversity(predicted_slates, self.number_of_movies)

        predicted_slates = predicted_slates.cpu()
        precision, hr = precision_hit_ratio(predicted_slates, ground_truth_slates)

        return precision, hr, diversity

    def run_experiment(self):
        total_losses = {"loss": [], "precision": [], "hr": [],
                        "diversity": [], "curr_epoch": []}

        for epoch_idx in range(self.starting_epoch, self.configs['num_of_epochs']):
            print(f"Epoch: {epoch_idx}")
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
