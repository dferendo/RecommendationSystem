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
import torch.nn.functional as F


def GDPPLoss(phiFake, phiReal, backward=True):
    def compute_diversity(phi):
        phi = F.normalize(phi, p=2, dim=1)
        SB = torch.mm(phi, phi.t())
        eigVals, eigVecs = torch.symeig(SB, eigenvectors=True)
        return eigVals, eigVecs

    def normalize_min_max(eigVals):
        minV, maxV = torch.min(eigVals), torch.max(eigVals)
        if abs(minV - maxV) < 1e-10:
            return eigVals
        return (eigVals - minV) / (maxV - minV)

    fakeEigVals, fakeEigVecs = compute_diversity(phiFake)
    realEigVals, realEigVecs = compute_diversity(phiReal)

    # Scaling factor to make the two losses operating in comparable ranges.
    magnitudeLoss = 0.0001 * F.mse_loss(target=realEigVals, input=fakeEigVals)
    structureLoss = -torch.sum(torch.mul(fakeEigVecs, realEigVecs), 0)
    normalizedRealEigVals = normalize_min_max(realEigVals)
    weightedStructureLoss = torch.sum(
        torch.mul(normalizedRealEigVals, structureLoss))
    gdppLoss = magnitudeLoss + weightedStructureLoss

    if backward:
        gdppLoss.backward(retain_graph=True)

    return gdppLoss.item()


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
        real_slates = real_slates.view(real_slates.shape[0], self.configs['slate_size'], self.num_of_movies)
        response_vector = values_to_unpack[4].to(self.device).float()

        '''    
        Update discriminator
        '''
        loss_dis = self.update_discriminator(real_slates, user_interactions_with_padding, number_of_interactions_per_user, response_vector)

        if idx != 0 and idx % self.CRITIC_ITERS == 0:
            for p in self.discriminator.parameters():
                p.requires_grad = False

            loss_gen = self.update_generator(real_slates, user_interactions_with_padding, number_of_interactions_per_user, response_vector)

            for p in self.discriminator.parameters():
                p.requires_grad = True
        else:
            loss_gen = None

        return loss_gen, loss_dis

    def eval_iteration(self, values_to_unpack):
        user_interactions_with_padding = values_to_unpack[0].to(self.device)
        number_of_interactions_per_user = values_to_unpack[1].to(self.device)
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

        dis_real = dis_real.mean()

        # Generate fake slates
        noise = torch.randn(user_interactions_with_padding.shape[0], self.configs['noise_hidden_dims'],
                            dtype=torch.float32, device=self.device)

        fake_slates = self.generator(user_interactions_with_padding, number_of_interactions_per_user, response_vector, noise)
        dis_fake, _ = self.discriminator(fake_slates.detach(), user_interactions_with_padding,
                                         number_of_interactions_per_user, response_vector)
        dis_fake = dis_fake.mean()

        # Calculate Gradient policy
        epsilon = torch.rand(real_slates.shape[0], 1, 1)
        epsilon = epsilon.expand(real_slates.size()).to(self.device)

        interpolation = epsilon * real_slates + ((1 - epsilon) * fake_slates)
        interpolation = torch.autograd.Variable(interpolation, requires_grad=True).to(self.device)

        dis_interpolates, _ = self.discriminator(interpolation, user_interactions_with_padding,
                                                 number_of_interactions_per_user, response_vector)
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

    def update_generator(self, real_slates, user_interactions_with_padding, number_of_interactions_per_user, response_vector):
        self.optimizer_gen.zero_grad()
        self.optimizer_dis.zero_grad()

        noise = torch.randn(user_interactions_with_padding.shape[0], self.configs['noise_hidden_dims'],
                            dtype=torch.float32, device=self.device)

        fake_slates = self.generator(user_interactions_with_padding, number_of_interactions_per_user, response_vector, noise)
        fake_loss, fake_h = self.discriminator(fake_slates, user_interactions_with_padding, number_of_interactions_per_user, response_vector)
        fake_loss = fake_loss.mean()
        fake_loss.backward(retain_graph=True)

        _, real_h = self.discriminator(real_slates, user_interactions_with_padding, number_of_interactions_per_user, response_vector)
        # gdpp_loss = GDPPLoss(real_h, fake_h, backward=True)

        # g_loss = -fake_loss + gdpp_loss
        g_loss = -fake_loss
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

    response_vector_dims = 1
    num_of_movies = train_loader.dataset.number_of_movies

    generator = Generator(num_of_movies, configs['slate_size'], response_vector_dims, configs['embed_dims'],
                          configs['noise_hidden_dims'], 512, configs['train_batch_size'])

    discriminator = Discriminator(512, configs['slate_size'], num_of_movies, configs['embed_dims'], response_vector_dims)

    experiment_builder = FullyConnectedGANExperimentBuilder(generator, discriminator, train_loader, test_loader,
                                                            num_of_movies, configs, print_learnable_parameters=True)
    experiment_builder.run_experiment()


if __name__ == '__main__':
    experiments_run()
