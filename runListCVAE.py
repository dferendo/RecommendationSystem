from utils.arg_parser import extract_args_from_json
from utils.data_provider import split_dataset
from utils.reset_seed import set_seeds
from utils.SlateFormation import generate_slate_formation
from utils.experiment_builder_GANs import ExperimentBuilderGAN
from dataloaders.SlateFormation import SlateFormationDataLoader, UserConditionedDataLoader
from models.ListCVAE import ListCVAE

import torch
import os
import pandas as pd
from torch.utils.data import DataLoader
from utils.experiment_builder import ExperimentBuilderNN


class ListCVAEExperimentBuilder(ExperimentBuilderNN):
    criterion = torch.nn.CrossEntropyLoss()

    def loss_function(self, recon_slates, slates, prior_mu, prior_log_variance):
        recon_slates = recon_slates.view(recon_slates.shape[0] * recon_slates.shape[1], recon_slates.shape[2])
        slates = slates.view(slates.shape[0] * slates.shape[1])

        entropy_loss = self.criterion(recon_slates, slates)
        KLD = -0.5 * torch.sum(1 + prior_log_variance - prior_mu.pow(2) - prior_log_variance.exp())
        return entropy_loss + KLD

    def pre_epoch_init_function(self):
        pass

    def train_iteration(self, idx, values_to_unpack):
        user_interactions_with_padding = values_to_unpack[1].to(self.device)
        number_of_interactions_per_user = values_to_unpack[2].to(self.device)
        real_slates = values_to_unpack[3].long().to(self.device)
        response_vector = values_to_unpack[4].float().to(self.device)

        decoder_out, mu, log_variance = self.model(real_slates, user_interactions_with_padding,
                                                   number_of_interactions_per_user, response_vector)

        loss = self.loss_function(decoder_out, real_slates, mu, log_variance)

        return loss

    def eval_iteration(self, values_to_unpack):
        user_interactions_with_padding = values_to_unpack[0].to(self.device)
        number_of_interactions_per_user = values_to_unpack[1].to(self.device)
        response_vector = torch.full((self.configs['test_batch_size'], self.configs['slate_size']), 1,
                                     device=self.device, dtype=torch.float32)

        slates = self.model.inference(user_interactions_with_padding, number_of_interactions_per_user, response_vector)

        return slates


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

    test_dataset = UserConditionedDataLoader(df_test, df_test_matrix, df_train, df_train_matrix, slate_formation)
    test_loader = DataLoader(test_dataset, batch_size=configs['test_batch_size'], shuffle=True, num_workers=4,
                             drop_last=True)

    return train_loader, test_loader, df_train_matrix


def experiments_run():
    configs = extract_args_from_json()
    print(configs)
    set_seeds(configs['seed'])

    train_loader, test_loader, df_train_matrix = get_data_loaders(configs)
    response_vector_dims = 1

    device = torch.device("cuda")

    model = ListCVAE(train_loader.dataset.number_of_movies, configs['slate_size'], response_vector_dims, configs['embed_dims'],
                     configs['encoder_dims'], configs['latent_dims'], configs['decoder_dims'], device)

    print(model)

    total_movies = len(df_train_matrix.columns)
    total_users = len(df_train_matrix.index)

    experiment_builder = ListCVAEExperimentBuilder(model, train_loader, test_loader, total_movies, configs, print_learnable_parameters=False)
    experiment_builder.run_experiment()


if __name__ == '__main__':
    experiments_run()
