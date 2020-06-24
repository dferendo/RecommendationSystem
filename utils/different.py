import torch, time, os, pickle
import torch.nn as nn
import numpy as np
import torch.optim as optim
import random
import math
import logging
import tqdm
import copy

from spotlight.losses import (adaptive_hinge_loss, bpr_loss, hinge_loss, pointwise_loss)
from spotlight.factorization.representations import BilinearNet
from spotlight.evaluation import precision_recall_score_slates, precision_recall_slates_atk
from spotlight.torch_utils import cpu, gpu, minibatch, set_seed, shuffle
import json

logging.basicConfig(format='%(message)s', level=logging.INFO)


class CGAN(object):
    def __init__(self, G=None,
                 D=None,
                 z_dim=100,
                 n_iter=15,
                 batch_size=128,
                 loss_fun='bce',
                 learning_rate=1e-4,
                 slate_size=3,
                 G_optimizer_func=None,
                 embedding_dim=5,
                 hidden_layer=16,
                 D_optimizer_func=None,
                 experiment_name="CGANs",
                 use_cuda=False,
                 random_state=None):

        self.exeriment_name = experiment_name
        self.experiment_folder = os.path.abspath("experiments_results/" + experiment_name)
        self.experiment_logs = os.path.abspath(os.path.join(self.experiment_folder, "result_outputs"))
        self.experiment_saved_models = os.path.abspath(os.path.join(self.experiment_folder, "saved_models"))

        self.starting_epoch = 0
        self.training = True
        if not os.path.exists("experiments_results"):  # If experiment directory does not exist
            os.mkdir("experiments_results")

        if not os.path.exists(self.experiment_folder):  # If experiment directory does not exist
            os.mkdir(self.experiment_folder)  # create the experiment directory

        if not os.path.exists(self.experiment_logs):
            os.mkdir(self.experiment_logs)  # create the experiment log directory

        if not os.path.exists(self.experiment_saved_models):
            os.mkdir(self.experiment_saved_models)  # create the experiment saved models directory

        self._n_iter = n_iter
        self.G = G
        self.slate_size = slate_size
        self.D = D
        self._learning_rate = learning_rate
        self._use_cuda = use_cuda
        self.G_optimizer_func = G_optimizer_func
        self.D_optimizer_func = D_optimizer_func
        self._random_state = random_state or np.random.RandomState()
        self.use_cuda = use_cuda
        self.hidden_layer = hidden_layer
        self.embedding_dim = embedding_dim
        self.z_dim = z_dim
        self.loss_fun = loss_fun
        self.weight_cliping_limit = 0.01
        self.n_critic = 5
        self._batch_size = batch_size
        self.logistic = nn.Sigmoid()
        self.chosen_epoch = - 1
        self.best_model = None
        self.best_precision = -1
        self.gamma = 10
        if use_cuda:
            self.device = torch.device('cuda')
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor
            self.device = torch.device('cpu')

        assert loss_fun in ('mse', 'bce')

    def _initialize(self):
        '''
            Initialize the networks and the optimizers and
            send them to cuda, if available. Save the configuration in a .json file
        '''
        self.G = self.G.to(self.device)
        self.D = self.D.to(self.device)

        if self.loss_fun == 'mse':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.MSELoss()

        self.G_optimizer = self.G_optimizer_func(
            self.G.parameters(),
            weight_decay=0,
            lr=self._learning_rate
        )
        self.D_optimizer = self.D_optimizer_func(
            self.D.parameters(),
            weight_decay=0,
            lr=self._learning_rate
        )
        self.embedding_layer = nn.Embedding(
            self.num_items + 1,
            self.embedding_dim,
            padding_idx=self.num_items)
        configuration = {
            'batch_size': self._batch_size,
            'z_dim': self.z_dim,
            'slate_size': self.slate_size,
            'n_iter': self._n_iter,
            'learning_rate': self._learning_rate,
            'users': self.num_users,
            'movies': self.num_items,
            'hidden_layer': self.hidden_layer,
            'embedding_dim': self.embedding_dim
        }
        with open(os.path.join(self.experiment_logs, 'configuration.json'), 'w') as fp:
            json.dump(configuration, fp)

    def one_hot_encoding(self, slates, num_items):

        '''
            Given an slate of size k, turns each item of the slate into 1-hot encoding in order to be fed into the discriminator.

        Parameters
        ----------
            slates: numpy array (users, self.k)
            num_items: number of avaialable items
        '''

        one_hot = torch.empty(0, slates.shape[1] * num_items).type(self.dtype)
        for z in slates:
            single_one_hot = nn.functional.one_hot(z.to(torch.int64), num_classes=num_items).type(self.dtype)
            single_one_hot = single_one_hot.reshape(1, -1)
            one_hot = torch.cat((one_hot, single_one_hot), 0)
        return one_hot

    def preprocess_train(self, interactions):
        '''
        Turn a sparse crs format array of interactions into torch.Tensor, which will be used to index the embedding layer.

        Parameters
        ----------
            interactions: sparse matrix contatining interactions

        Output
        ---------
            valid_rows: np.array
                    containing which users in the training have indeed available training interactions
            pad_sequence: torch.Tensor (valid_users, max interactions)
                torch.Tensor for indexing the embedding layer in batches. If a user has less than max_interactions then he is padded until he reached max_interactions
                Padding_value: self.num_items. Indexes the embedding layer with zero vector

        '''

        row, col = interactions.nonzero()
        valid_rows = np.unique(row)
        indices = np.where(row[:-1] != row[1:])
        indices = indices[0] + 1
        vec = np.split(col, indices)
        vec = [torch.Tensor(x) for x in vec]
        return valid_rows, torch.nn.utils.rnn.pad_sequence(vec, batch_first=True, padding_value=self.num_items)

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def fit(self,train_vec,train_slates, users, movies,valid_vec,valid_cold_users,valid_set):
        self.num_users = users
        self.num_items = movies

        self._initialize()

        # Counter which counts the updates of generator and the discriminator.
        steps_performed = 0

        train_vec = train_vec.type(self.dtype)
        valid_vec = valid_vec.type(self.dtype)

        user_slate_tensor = torch.from_numpy(train_slates).type(self.dtype)
        logging.info('training start!!')

        # Dictionary to store logs.

        total_losses = {"G_loss": [], "D_loss": [], 'G_pre': [], 'G_rec': [], "curr_epoch": [], 'Val_prec': []}

        for epoch_num in range(self._n_iter):

            recall = []
            g_train_epoch_loss = []
            d_train_epoch_loss = []

            current_epoch_losses = {"G_loss": [], "D_loss": [], 'G_pre': [], 'G_rec': [], 'Val_prec': []}

            with tqdm.tqdm(total=train_slates.shape[0]) as pbar_train:

                # TRAINING
                for batch_user, batch_slate in minibatch(train_vec, user_slate_tensor, batch_size=self._batch_size):

                    steps_performed += 1
                    d_loss = self.train_discriminator_iteration(batch_user, batch_slate)
                    d_train_epoch_loss.append(d_loss)

                    # Update generator once every 5 update of discriminator
                    if steps_performed % self.n_critic == 0:
                        g_loss, precision_batch, recall_batch = self.train_generator_iteration(batch_user, batch_slate)
                        g_train_epoch_loss.append(g_loss)
                        current_epoch_losses["G_loss"].append(g_loss)
                        current_epoch_losses["D_loss"].append(d_loss)
                        current_epoch_losses["G_pre"] += precision_batch
                        current_epoch_losses["G_rec"] += recall_batch
                    pbar_train.update(self._batch_size)

                # VALIDATION ITERATION
                self.G.eval()
                results_dict = self.test(valid_vec, valid_set, valid_cold_users)
                logging.info(str(results_dict['precision']))
                if results_dict['precision'] > self.best_precision:
                    self.best_model = copy.deepcopy(self.G)
                    self.best_precision = results_dict['precision']
                    self.chosen_epoch = epoch_num
                current_epoch_losses["Val_prec"].append(results_dict['precision'])

                total_losses['curr_epoch'].append(epoch_num)
                for key, value in current_epoch_losses.items():
                    total_losses[key].append(np.mean(value))

            g_train_epoch_loss = np.mean(g_train_epoch_loss)
            d_train_epoch_loss = np.mean(d_train_epoch_loss)

            logging.info("--------------- Epoch %d ---------------" % epoch_num)
            logging.info("G_Loss: {}".format(g_train_epoch_loss))
            logging.info("D_Loss: {} D(x): {}".format(d_train_epoch_loss, self.sigmoid(d_train_epoch_loss)))

        # Select model with best validation accuracy
        self.G = self.best_model
        logging.info('Model chosen from: {}'.format(self.chosen_epoch))
        try:
            state_dict_G = self.G.module.state_dict()
        except AttributeError:
            state_dict_G = self.G.state_dict()
        self.save_readable_model(self.experiment_saved_models, state_dict_G)
        self.training = False

    def gradient_penalty(self, data, generated_data, user_condition, gamma=10):

        batch_size = data.size(0)
        epsilon = torch.rand(batch_size, 1)
        epsilon = epsilon.expand_as(data)

        epsilon = epsilon.to(self.device)

        interpolation = epsilon * data.data + (1 - epsilon) * generated_data.data
        interpolation = torch.autograd.Variable(interpolation, requires_grad=True)
        interpolation = interpolation.to(self.device)

        interpolation_logits = self.D(interpolation, user_condition)
        grad_outputs = torch.ones(interpolation_logits.size())

        if self.use_cuda:
            grad_outputs = grad_outputs.cuda()

        gradients = torch.autograd.grad(outputs=interpolation_logits,
                                        inputs=interpolation,
                                        only_inputs=True,
                                        grad_outputs=grad_outputs,
                                        create_graph=True,
                                        retain_graph=True)[0]

        gradients = gradients.view(batch_size, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-16)
        return self.gamma * ((gradients_norm - 1) ** 2).mean()

    def train_generator_iteration(self, batch_user, batch_slate):

        '''
        Function to perform an update to the generator

            Parameters
            ----------
                batch_user: instance of torch.Tensor
                    user_reprenation fed into the generator
                batch_slate: instance of torch.Tensor
                Target slates which the generator must predict given the past interactions (train_vec)
            Output
        ---------
        '''

        for p in self.D.parameters():
            p.requires_grad = False

        self.G_optimizer.zero_grad()

        # Sample random noise
        z = torch.rand(batch_user.shape[0], self.z_dim, device=self.device).type(self.dtype)
        fake_slates = self.G(z, batch_user)
        fake_slates = torch.cat(fake_slates, dim=-1)

        d_fake_val = self.D(fake_slates, batch_user)
        g_loss = -d_fake_val.mean()

        g_loss.backward()
        self.G_optimizer.step()

        # Get training precision and recall
        self.G.eval()
        fake_slates_result = self.G(z, batch_user, inference=True)
        precision, recall = precision_recall_slates_atk(fake_slates_result.type(torch.int64), batch_slate,
                                                        k=self.slate_size)
        self.G.train()
        return g_loss.item(), precision, recall

    def train_discriminator_iteration(self, batch_user, batch_slate):
        '''
        Function to perform an update to the discriminator

            Parameters
            ----------
                batch_user: instance of torch.Tensor
                    user_reprenation fed into the generator
                batch_slate: instance of torch.Tensor
                Target slates which the generator must predict given the past interactions (train_vec)
            Output
        ---------
        '''

        self.G.train()
        self.D.train()

        z = torch.rand(batch_user.shape[0], self.z_dim, device=self.device).type(self.dtype)

        # Set gradients equal to true, to perform an update. Disabled during generator training
        for p in self.D.parameters():
            p.requires_grad = True

        self.D_optimizer.zero_grad()

        # Clip the weights to ensure Lipschitz constraint

        for p in self.D.parameters():
            p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)

        real_slates = self.one_hot_encoding(batch_slate, self.num_items)
        d_real_val = self.D(real_slates, batch_user)
        d_loss_real = d_real_val.mean()

        fake_slates = self.G(z, batch_user)
        fake_slates = torch.cat(fake_slates, dim=-1)

        d_fake_val = self.D(fake_slates.detach(), batch_user)
        d_loss_fake = d_fake_val.mean()

        d_loss = d_loss_fake - d_loss_real
        d_loss.backward()

        self.D_optimizer.step()

        return d_loss.item()

    def run_val_iteration(self, batch_user, batch_slate):

        '''
        Function to evaluate the generator's precision and the networks losses. Used to monitor the training

            Parameters
            ----------
                batch_user: instance of torch.Tensor
                    user_reprenation fed into the generator
                batch_slate: instance of torch.Tensor
                Target slates which the generator must predict given the past interactions (train_vec)
            Output
        ---------
        '''

        self.G.eval()
        self.D.eval()

        # No loner required under the new Wassertein objective

        # valid = (torch.ones(batch_user.shape[0], 1) * 0.9).type(self.dtype)
        # fake = (torch.zeros(batch_user.shape[0], 1)).type(self.dtype)

        z = torch.rand(batch_user.shape[0], self.z_dim, device=self.device).type(self.dtype)

        ## Test discriminator on real images
        real_slates = self.one_hot_encoding(batch_slate, self.num_items)
        fake_slates = self.G(z, batch_user)
        fake_slates = torch.cat(fake_slates, dim=-1)

        d_real_val = self.D(real_slates, batch_user)
        real_score = d_real_val.mean().item()

        # Test discriminator on fake images
        d_fake_val = self.D(fake_slates.detach(), batch_user)

        real_loss = self.criterion(d_real_val, valid)
        fake_loss = self.criterion(d_fake_val, fake)
        d_loss = fake_loss + real_loss

        d_fake_val = self.D(fake_slates, batch_user)
        fake_score = d_fake_val.mean().item()
        g_loss = self.criterion(d_fake_val, valid)

        return g_loss.item(), d_loss.item(), fake_score, real_score

    def test(self, train_vec, test, cold_start_users=None):

        '''
        Main function of our framework. Used to train the generator and the discriminator jointly.

        Parameters
        ----------
            train_vec: instance of torch.Tensor
                User history in torch.Tensor used to index the embedding layer of the generator and the discriminator
            cold_start_users: instance of torch.Tensor
                Users with no previous interactions will use a different index (the padded one)
            test: Sparse matrix in csr format
                Used to evaluate the performance of the validation set.

        Output
        ---------
            test_results: instance of python dictionary
                Python dictionary containing precision and recall of generator on the test set at self.k
        '''

        self.G.eval()
        total_losses = {"precision": [], "recall": []}
        train_vec = train_vec.to(self.device)
        for minibatch_num, user_batch in enumerate(minibatch(train_vec, batch_size=self._batch_size)):
            z = torch.rand(user_batch.shape[0], self.z_dim, device=self.device).type(self.dtype)

            slates = self.G(z, user_batch, inference=True)
            precision, recall = precision_recall_score_slates(slates.type(torch.int64), test[minibatch_num *
                                                                                             user_batch.shape[
                                                                                                 0]: minibatch_num *
                                                                                                     user_batch.shape[
                                                                                                         0] +
                                                                                                     user_batch.shape[
                                                                                                         0], :],
                                                              k=self.slate_size)
            print(slates)
            total_losses["precision"] += precision
            total_losses["recall"] += recall

        # Test generator on cold start users
        if cold_start_users != None:
            # Create special embedding for cold-start users, a (cold_start_users,self.gan_embeeding_dim) torch.Tensor
            cold_start_users_tensor = torch.empty((cold_start_users.shape[0], self.embedding_dim)).fill_(
                self.num_items).type(self.dtype)
            for minibatch_num, user_batch in enumerate(minibatch(cold_start_users_tensor, batch_size=self._batch_size)):
                z = torch.rand(user_batch.shape[0], self.z_dim, device=self.device).type(self.dtype)
                slates = self.G(z, user_batch, inference=True)

                precision, recall = precision_recall_score_slates(slates.type(torch.int64), cold_start_users[
                                                                                            minibatch_num *
                                                                                            user_batch.shape[
                                                                                                0]: minibatch_num *
                                                                                                    user_batch.shape[
                                                                                                        0] +
                                                                                                    user_batch.shape[0],
                                                                                            :], k=self.slate_size)
                total_losses["precision"] += precision
                total_losses["recall"] += recall

        test_results = {
            'precision': np.mean(total_losses["precision"]),
            'recall': np.mean(total_losses["recall"]),
            'at': self.slate_size
        }

        with open(os.path.join(self.experiment_logs, 'test_results.json'), 'w') as fp:
            json.dump(test_results, fp)

        return test_results

    def save_readable_model(self, model_save_dir, state_dict):
        state = {'network': state_dict}  # save network parameter and other variables.
        fname = os.path.join(model_save_dir, "generator")
        logging.info('Saving state in {}'.format(fname))
        torch.save(state, f=fname)  # save state at prespecified filepath





