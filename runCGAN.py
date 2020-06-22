from utils.arg_parser import extract_args_from_json
from utils.data_provider import split_dataset
from utils.reset_seed import set_seeds
from utils.SlateFormation import generate_slate_formation, generate_test_slate_formation
from utils.experiment_builder_GANs import ExperimentBuilderGAN
from dataloaders.SlateFormation import SlateFormationDataLoader, SlateFormationTestDataLoader
from models.CGAN import Generator, Discriminator

import torch
import os
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable


class FullyConnectedGANExperimentBuilder(ExperimentBuilderGAN):
    CRITIC_ITERS = 5
    LAMBDA = 10
    adversarial_loss = torch.nn.BCELoss()

    def pre_epoch_init_function(self):
        pass

    def loss_function(self, values):
        pass

    def train_iteration(self, idx, values_to_unpack):
        user_interactions_with_padding = values_to_unpack[1].to(self.device)
        number_of_interactions_per_user = values_to_unpack[2].to(self.device)
        real_slates = values_to_unpack[3].to(self.device).float()
        response_vector = values_to_unpack[4].to(self.device).float()

        '''    
        Update discriminator
        '''
        loss_dis = self.update_discriminator(real_slates, user_interactions_with_padding, number_of_interactions_per_user, response_vector)

        # if idx != 0 and idx % self.CRITIC_ITERS == 0:
        #     for p in self.discriminator.parameters():
        #         p.requires_grad = False
        #
        loss_gen = self.update_generator(real_slates, user_interactions_with_padding, number_of_interactions_per_user, response_vector)

            # for p in self.discriminator.parameters():
            #     p.requires_grad = True
        # else:
        #     loss_gen = None

        return loss_gen, loss_dis

    def eval_iteration(self, values_to_unpack):
        user_interactions_with_padding = values_to_unpack[1].to(self.device)
        number_of_interactions_per_user = values_to_unpack[2].to(self.device)
        response_vector = torch.full((self.configs['test_batch_size'], self.configs['slate_size']), 1,
                                     device=self.device, dtype=torch.float32)

        noise = torch.randn(user_interactions_with_padding.shape[0], self.configs['noise_hidden_dims'],
                            dtype=torch.float32, device=self.device)

        fake_slates = self.generator(user_interactions_with_padding, number_of_interactions_per_user, response_vector, noise,
                                     inference=True)

        return fake_slates

    def update_discriminator(self, real_slates, user_interactions_with_padding, number_of_interactions_per_user, response_vector):
        self.optimizer_dis.zero_grad()

        dis_real, _ = self.discriminator(real_slates, user_interactions_with_padding, number_of_interactions_per_user, response_vector)

        # Generate fake slates
        noise = torch.randn(user_interactions_with_padding.shape[0], self.configs['noise_hidden_dims'],
                            dtype=torch.float32, device=self.device)

        fake_slates = self.generator(user_interactions_with_padding, number_of_interactions_per_user, response_vector, noise)
        print(fake_slates)

        dis_fake, _ = self.discriminator(fake_slates.detach(), user_interactions_with_padding,
                                         number_of_interactions_per_user, response_vector)

        # Adversarial ground truths
        valid = Variable(torch.Tensor(real_slates.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(torch.Tensor(real_slates.size(0), 1).fill_(0.0), requires_grad=False)

        real_loss = self.adversarial_loss(dis_real, valid)
        fake_loss = self.adversarial_loss(dis_fake, fake)

        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        self.optimizer_dis.step()

        return d_loss

    def update_generator(self, real_slates, user_interactions_with_padding, number_of_interactions_per_user, response_vector):
        self.optimizer_gen.zero_grad()
        self.optimizer_dis.zero_grad()

        noise = torch.randn(user_interactions_with_padding.shape[0], self.configs['noise_hidden_dims'],
                            dtype=torch.float32, device=self.device)

        fake_slates = self.generator(user_interactions_with_padding, number_of_interactions_per_user, response_vector, noise)
        fake_loss, fake_h = self.discriminator(fake_slates, user_interactions_with_padding, number_of_interactions_per_user, response_vector)
        valid = Variable(torch.Tensor(real_slates.size(0), 1).fill_(1.0), requires_grad=False)
        g_loss = self.adversarial_loss(fake_loss, valid)
        # gdpp_loss = GDPPLoss(real_h, fake_h, backward=True)

        # g_loss = -fake_loss + gdpp_loss
        g_loss.backward()
        self.optimizer_gen.step()
        return g_loss


def get_data_loaders(configs):
    slate_formation_file_name = 'sf_{}_{}_{}.csv'.format(configs['slate_size'],
                                                         '-'.join(str(e) for e in configs['negative_sampling_for_slates']),
                                                         configs['is_training'])
    slate_formation_file_location = os.path.join(configs['data_location'], slate_formation_file_name)

    slate_formation_file_name = 'sf_{}_{}_{}_test.csv'.format(configs['slate_size'],
                                                              '-'.join(str(e) for e in configs['negative_sampling_for_slates']),
                                                              configs['is_training'])

    slate_formation_test_file_location = os.path.join(configs['data_location'], slate_formation_file_name)

    # df_train, df_test, df_train_matrix, df_test_matrix, movies_categories = split_dataset(configs)

    # Check if we have the slates for training
    if os.path.isfile(slate_formation_file_location) and os.path.isfile(slate_formation_test_file_location):
        slate_formation = pd.read_csv(slate_formation_file_location)
        test_slate_formation = pd.read_csv(slate_formation_test_file_location)
    else:
        slate_formation = generate_slate_formation(df_train, df_train_matrix, configs['slate_size'],
                                                   configs['negative_sampling_for_slates'],
                                                   slate_formation_file_location)

        test_slate_formation = generate_test_slate_formation(df_test, df_train, df_train_matrix,
                                                             slate_formation_test_file_location)

    train_dataset = SlateFormationDataLoader(slate_formation, 8)
    train_loader = DataLoader(train_dataset, batch_size=configs['train_batch_size'], shuffle=False, num_workers=4,
                              drop_last=True)

    test_dataset = SlateFormationTestDataLoader(test_slate_formation, 8)
    test_loader = DataLoader(test_dataset, batch_size=configs['test_batch_size'], shuffle=True, num_workers=4,
                             drop_last=True)

    return train_loader, test_loader


def experiments_run():
    configs = extract_args_from_json()
    print(configs)
    set_seeds(configs['seed'])

    train_loader, test_loader = get_data_loaders(configs)

    print('number of movies: ', train_loader.dataset.number_of_movies)

    response_vector_dims = 1

    generator = Generator(train_loader.dataset.number_of_movies, configs['slate_size'], configs['embed_dims'],
                          configs['noise_hidden_dims'], configs['hidden_layers_dims_gen'], response_vector_dims)

    discriminator = Discriminator(train_loader.dataset.number_of_movies, configs['slate_size'], configs['embed_dims'],
                                  configs['hidden_layers_dims_dis'], response_vector_dims)

    experiment_builder = FullyConnectedGANExperimentBuilder(generator, discriminator, train_loader, test_loader, configs,
                                                            print_learnable_parameters=True)
    experiment_builder.run_experiment()


if __name__ == '__main__':
    experiments_run()
