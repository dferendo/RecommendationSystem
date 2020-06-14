from utils.arg_parser import extract_args_from_json
from utils.data_provider import split_dataset
from utils.reset_seed import set_seeds
from utils.SlateFormation import generate_slate_formation
from utils.experiment_builder_GANs import ExperimentBuilderGAN
from dataloaders.SlateFormation import SlateFormationDataLoader, UserConditionedDataLoader
from models.CGAN import Generator, Discriminator

import torch
import os
import pandas as pd
from torch.utils.data import DataLoader


class FullyConnectedGANExperimentBuilder(ExperimentBuilderGAN):
    CRITIC_ITERS = 5
    LAMBDA = 10

    def pre_epoch_init_function(self):
        pass

    def loss_function(self, values):
        pass

    def train_iteration(self, idx, values_to_unpack):
        user_interactions_with_padding = values_to_unpack[1].to(self.device)
        number_of_interactions_per_user = values_to_unpack[2].to(self.device)
        real_slates = values_to_unpack[3].to(self.device).float()

        '''    
        Update discriminator
        '''
        loss_dis = self.update_discriminator(real_slates, user_interactions_with_padding, number_of_interactions_per_user)

        if idx != 0 and idx % self.CRITIC_ITERS == 0:
            for p in self.discriminator.parameters():
                p.requires_grad = False

            loss_gen = self.update_generator(user_interactions_with_padding, number_of_interactions_per_user)

            for p in self.discriminator.parameters():
                p.requires_grad = True
        else:
            loss_gen = None

        return loss_gen, loss_dis

    def eval_iteration(self, values_to_unpack):
        user_interactions_with_padding = values_to_unpack[0].to(self.device)
        number_of_interactions_per_user = values_to_unpack[1].to(self.device)

        noise = torch.randn(user_interactions_with_padding.shape[0], self.configs['noise_hidden_dims'],
                            dtype=torch.float32, device=self.device)

        fake_slates = self.generator(user_interactions_with_padding, number_of_interactions_per_user, noise,
                                     inference=True)

        return fake_slates

    def update_discriminator(self, real_slates, user_interactions_with_padding, number_of_interactions_per_user):
        self.discriminator.zero_grad()

        dis_real = self.discriminator(real_slates, user_interactions_with_padding, number_of_interactions_per_user)
        dis_real = dis_real.mean()

        # Generate fake slates
        noise = torch.randn(user_interactions_with_padding.shape[0], self.configs['noise_hidden_dims'],
                            dtype=torch.float32, device=self.device)

        fake_slates = self.generator(user_interactions_with_padding, number_of_interactions_per_user, noise)
        dis_fake = self.discriminator(fake_slates.detach(), user_interactions_with_padding,
                                      number_of_interactions_per_user)
        dis_fake = dis_fake.mean()

        # Calculate Gradient policy
        epsilon = torch.rand(real_slates.shape[0], 1)
        epsilon = epsilon.expand(real_slates.size()).to(self.device)

        interpolation = epsilon * real_slates + ((1 - epsilon) * fake_slates)
        interpolation = torch.autograd.Variable(interpolation, requires_grad=True).to(self.device)

        dis_interpolates = self.discriminator(interpolation, user_interactions_with_padding,
                                              number_of_interactions_per_user)
        grad_outputs = torch.ones(dis_interpolates.size()).to(self.device)

        gradients = torch.autograd.grad(outputs=dis_interpolates,
                                        inputs=interpolation,
                                        grad_outputs=grad_outputs,
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA

        d_loss = dis_real - dis_fake + gradient_penalty
        d_loss.backward()

        self.optimizer_dis.step()

        return d_loss

    def update_generator(self, user_interactions_with_padding, number_of_interactions_per_user):
        noise = torch.randn(user_interactions_with_padding.shape[0], self.configs['noise_hidden_dims'],
                            dtype=torch.float32, device=self.device)

        slates = self.generator(user_interactions_with_padding, number_of_interactions_per_user, noise)
        loss = self.discriminator(slates, user_interactions_with_padding, number_of_interactions_per_user)
        loss = loss.mean()
        loss.backward()
        g_loss = -loss

        self.optimizer_gen.step()
        return g_loss


def get_data_loaders(configs):
    df_train, df_test, df_train_matrix, df_test_matrix, movies_categories = split_dataset(configs)

    slate_formation_file_name = 'sf_{}_{}_{}.csv'.format(configs['slate_size'],
                                                         '-'.join(
                                                             str(e) for e in configs['negative_sampling_for_slates']),
                                                         configs['is_training'])
    slate_formation_file_location = os.path.join(configs['data_location'], slate_formation_file_name)

    # Check if we have the slates for training
    if os.path.isfile(slate_formation_file_location):
        slate_formation = pd.read_csv(slate_formation_file_location)
    else:
        slate_formation = generate_slate_formation(df_train, df_train_matrix, configs['slate_size'],
                                                   configs['negative_sampling_for_slates'],
                                                   slate_formation_file_location)

    train_dataset = SlateFormationDataLoader(slate_formation, df_train_matrix)
    train_loader = DataLoader(train_dataset, batch_size=configs['train_batch_size'], shuffle=True, num_workers=4,
                              drop_last=True)

    test_dataset = UserConditionedDataLoader(df_test, df_test_matrix, df_train, df_train_matrix)
    test_loader = DataLoader(test_dataset, batch_size=configs['test_batch_size'], shuffle=True, num_workers=4,
                             drop_last=True)

    return train_loader, test_loader


def experiments_run():
    configs = extract_args_from_json()
    print(configs)
    set_seeds(configs['seed'])

    train_loader, test_loader = get_data_loaders(configs)

    generator = Generator(train_loader.dataset.number_of_movies, configs['slate_size'], configs['embed_dims'],
                          configs['noise_hidden_dims'], configs['hidden_layers_dims_gen'])

    discriminator = Discriminator(train_loader.dataset.number_of_movies, configs['slate_size'], configs['embed_dims'],
                                  configs['hidden_layers_dims_dis'])

    experiment_builder = FullyConnectedGANExperimentBuilder(generator, discriminator, train_loader, test_loader, configs,
                                                            print_learnable_parameters=True)
    experiment_builder.run_experiment()


if __name__ == '__main__':
    experiments_run()
