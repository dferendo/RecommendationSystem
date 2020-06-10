from utils.arg_parser import extract_args_from_json
from utils.data_provider import split_dataset, load_movie_categories
from utils.reset_seed import set_seeds
from utils.SlateFormation import generate_slate_formation
from dataloaders.SlateFormation import SlateFormationDataLoader
from torch.utils.data import DataLoader
import os
import pandas as pd
from utils.experiment_builder import ExperimentBuilder


class GreedyMLPExperimentBuilder(ExperimentBuilder):

    def pre_epoch_init_function(self):
        pass

    def loss_function(self, values):
        pass

    def forward_model_training(self, values_to_unpack):
        pass

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

    return train_loader


def experiments_run():
    configs = extract_args_from_json()
    set_seeds(configs['seed'])

    train_loader = get_data_loaders(configs)

    for values in train_loader:
        print(values)




if __name__ == '__main__':
    experiments_run()
    # batch_size = 2
    # num_of_movies = 5
    # slate_size = 6
    # embed_dims = 10
    # noise_hidden_dims = 20
    # hidden_layers_dims_gen = [20, 40, 60]
    # hidden_layers_dims_dis = [512, 512, 512]
    #
    # gen = Generator(num_of_movies, slate_size, embed_dims, noise_hidden_dims, hidden_layers_dims_gen)
    #
    # slates = torch.LongTensor([[0, 1, 2, 3, 4, 4], [0, 0, 0, 1, 1, 1]]).cuda()
    # user_interactions_with_padding = torch.LongTensor([[2, 3, 4], [0, 1, 5]]).cuda()
    # number_of_interactions_per_user = torch.LongTensor([[3], [2]]).cuda()
    #
    # for epoch in range(5):
    #     dis = Discriminator(num_of_movies, slate_size, embed_dims, hidden_layers_dims_dis)
    #
    #     # Loss functions
    #     criterion = torch.nn.BCELoss()
    #
    #     if torch.cuda.is_available():
    #         gen.cuda()
    #         dis.cuda()
    #         criterion.cuda()
    #
    #     optimizer_G = torch.optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.5))
    #     optimizer_D = torch.optim.Adam(dis.parameters(), lr=0.0002, betas=(0.5, 0.5))
    #
    #     # Adversarial ground truths
    #     valid = torch.tensor(np.full((batch_size, 1), 1.0), dtype=torch.float32, requires_grad=False).cuda()
    #     fake = torch.tensor(np.full((batch_size, 1), 0.0), dtype=torch.float32, requires_grad=False).cuda()
    #
    #     optimizer_G.zero_grad()
    #
    #     noise = torch.normal(0, 1, size=(batch_size, noise_hidden_dims)).cuda()
    #
    #     gen_fake = gen(user_interactions_with_padding, number_of_interactions_per_user, noise)
    #     validity = dis(gen_fake, user_interactions_with_padding, number_of_interactions_per_user)
    #
    #     g_loss = criterion(validity, fake)
    #     g_loss.backward()
    #
    #     plot_grad_flow(gen.named_parameters())
    #
    #     optimizer_G.step()
    #
    #     '''
    #     Training Discriminator
    #     '''
    #     optimizer_D.zero_grad()
    #
    #     # Loss for real images
    #     validity_real = dis(slates, user_interactions_with_padding, number_of_interactions_per_user)
    #     d_real_loss = criterion(validity_real, valid)
    #
    #     # Loss for fake images
    #     validity_fake = dis(gen_fake.detach(), user_interactions_with_padding, number_of_interactions_per_user)
    #     d_fake_loss = criterion(validity_fake, fake)
    #
    #     # Total discriminator loss
    #     d_loss = (d_real_loss + d_fake_loss) / 2
    #
    #     d_loss.backward()
    #     optimizer_D.step()
