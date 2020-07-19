from utils.arg_parser import extract_args_from_json
from utils.reset_seed import set_seeds
from models.ListCVAE import ListCVAE, Parameters
from utils.slate_formation import get_data_loaders

import torch
from utils.experiment_builder_CVAE import ExperimentBuilderCVAE


def experiments_run():
    configs = extract_args_from_json()
    print(configs)
    set_seeds(configs['seed'])

    train_loader, test_loader, data_configs, movie_categories = get_data_loaders(configs, False)

    if configs['diverse']:
        # One dims maximize utility, one dim genres maximization
        response_vector_dims = 2
    else:
        response_vector_dims = 1

    device = torch.device("cuda")

    encoder_params = Parameters(configs['enc_batch_norm'], configs['enc_dropout'], configs['enc_act'])
    decoder_params = Parameters(configs['dec_batch_norm'], configs['dec_dropout'], configs['dec_act'])
    prior_params = Parameters(configs['prior_batch_norm'], configs['prior_dropout'], configs['prior_act'])

    gdpp_active = False

    if configs['gdpp_weight'] > 0:
        gdpp_active = True

    model = ListCVAE(train_loader.dataset.number_of_movies, configs['slate_size'], response_vector_dims, configs['embed_dims'],
                     configs['encoder_dims'], configs['latent_dims'], configs['decoder_dims'], configs['prior_dims'], device,
                     encoder_params, decoder_params, prior_params, gdpp_active)

    experiment_builder = ExperimentBuilderCVAE(model, train_loader, test_loader, data_configs['number_of_movies'],
                                               movie_categories,
                                               configs)
    experiment_builder.run_experiment()


if __name__ == '__main__':
    experiments_run()
