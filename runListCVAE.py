from utils.arg_parser import extract_args_from_json
from utils.reset_seed import set_seeds
from models.ListCVAE import ListCVAE
from utils.slate_formation import get_data_loaders

import torch
from utils.experiment_builder_CVAE import ExperimentBuilderCVAE


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

    experiment_builder = ExperimentBuilderCVAE(model, train_loader, test_loader, data_configs['number_of_movies'], configs)
    experiment_builder.run_experiment()


if __name__ == '__main__':
    experiments_run()
