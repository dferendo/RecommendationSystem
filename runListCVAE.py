from utils.arg_parser import extract_args_from_json
from utils.reset_seed import set_seeds
from models.ListCVAE import ListCVAE
from utils.slate_formation import get_data_loaders

import torch
from utils.experiment_builder import ExperimentBuilderNN


class ListCVAEExperimentBuilder(ExperimentBuilderNN):
    criterion = torch.nn.CrossEntropyLoss()

    def loss_function(self, recon_slates, slates, prior_mu, prior_log_variance):
        recon_slates = recon_slates.view(recon_slates.shape[0] * recon_slates.shape[1], recon_slates.shape[2])
        slates = slates.view(slates.shape[0] * slates.shape[1])

        entropy_loss = self.criterion(recon_slates, slates)
        KLD = -0.5 * torch.sum(1 + prior_log_variance - prior_mu.pow(2) - prior_log_variance.exp())

        return entropy_loss + (KLD * self.configs['beta_weight'])

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
        user_interactions_with_padding = values_to_unpack[1].to(self.device)
        number_of_interactions_per_user = values_to_unpack[2].to(self.device)
        response_vector = torch.full((user_interactions_with_padding.shape[0], self.configs['slate_size']), 1,
                                     device=self.device, dtype=torch.float32)

        slates = self.model.inference(user_interactions_with_padding, number_of_interactions_per_user, response_vector)

        return slates


def experiments_run():
    configs = extract_args_from_json()
    print(configs)
    set_seeds(configs['seed'])

    train_loader, test_loader, data_configs = get_data_loaders(configs, False)
    response_vector_dims = 1

    device = torch.device("cuda")

    model = ListCVAE(train_loader.dataset.number_of_movies, configs['slate_size'], response_vector_dims, configs['embed_dims'],
                     configs['encoder_dims'], configs['latent_dims'], configs['decoder_dims'], configs['prior_dims'], device)

    print(model)

    experiment_builder = ListCVAEExperimentBuilder(model, train_loader, test_loader, train_loader.dataset.number_of_movies, configs, print_learnable_parameters=False)
    experiment_builder.run_experiment()


if __name__ == '__main__':
    experiments_run()
