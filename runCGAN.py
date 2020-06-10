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
    # Loss functions
    criterion = torch.nn.BCELoss()
    # criterion = torch.nn.MSELoss()
    # criterion = torch.nn.BCEWithLogitsLoss()

    real_label = 1
    fake_label = 0

    def pre_epoch_init_function(self):
        pass

    def loss_function(self, values):
        pass

    def train_iteration(self, values_to_unpack):
        user_interactions_with_padding = values_to_unpack[1].to(self.device)
        number_of_interactions_per_user = values_to_unpack[2].to(self.device)
        real_slates = values_to_unpack[3].to(self.device).float()

        '''    
        Update discriminator: maximize log(D(x)) + log(1 - D(G(z)))
        '''
        self.optimizer_dis.zero_grad()

        output = self.discriminator(real_slates, user_interactions_with_padding, number_of_interactions_per_user)

        labels = torch.full((real_slates.shape[0], 1), self.real_label, device=self.device, dtype=torch.float32)
        error_real_dis = self.criterion(output, labels)
        error_real_dis.backward()

        noise = torch.randn(user_interactions_with_padding.shape[0], self.configs['noise_hidden_dims'],
                            dtype=torch.float32, device=self.device)

        fake_slates = self.generator(user_interactions_with_padding, number_of_interactions_per_user, noise)
        output = self.discriminator(fake_slates.detach(), user_interactions_with_padding, number_of_interactions_per_user)

        labels.fill_(self.fake_label)
        error_fake_dis = self.criterion(output, labels)
        error_fake_dis.backward()

        # Add the gradients from the all-real and all-fake batches
        loss_dis = float((error_real_dis + error_fake_dis).item())

        self.optimizer_dis.step()

        '''    
        Update generator: maximize log(D(G(z)))
        '''
        self.optimizer_gen.zero_grad()
        # Fake labels are real for generator cost
        labels.fill_(self.real_label)

        output = self.discriminator(fake_slates, user_interactions_with_padding, number_of_interactions_per_user)
        error_gen = self.criterion(output, labels)
        error_gen.backward()

        loss_gen = float(error_gen.item())

        self.optimizer_gen.step()

        return loss_gen, loss_dis

    def forward_model_test(self, values_to_unpack):
        pass


def get_data_loaders(configs):
    df_train, df_test, df_train_matrix, df_test_matrix, movies_categories = split_dataset(configs)

    slate_formation_file_name = 'sf_{}_{}_{}.csv'.format(configs['slate_size'],
                                                         '-'.join(
                                                             str(e) for e in configs['negative_sampling_for_slates']),
                                                         configs['is_training'])
    slate_formation_file_location = os.path.join(configs['save_location'], slate_formation_file_name)

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

    for i in test_dataset:
        pass

    return train_loader


def experiments_run():
    configs = extract_args_from_json()
    set_seeds(configs['seed'])

    train_loader = get_data_loaders(configs)

    generator = Generator(train_loader.dataset.number_of_movies, configs['slate_size'], configs['embed_dims'],
                          configs['noise_hidden_dims'], configs['hidden_layers_dims_gen'])

    discriminator = Discriminator(train_loader.dataset.number_of_movies, configs['slate_size'], configs['embed_dims'],
                                  configs['hidden_layers_dims_dis'])

    experiment_builder = FullyConnectedGANExperimentBuilder(generator, discriminator, train_loader, None, configs,
                                                            print_learnable_parameters=False)
    experiment_builder.run_experiment()


if __name__ == '__main__':
    experiments_run()
