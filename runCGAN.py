from utils.arg_parser import extract_args_from_json
from utils.reset_seed import set_seeds
from utils.experiment_builder_GANs import ExperimentBuilderGAN
from models.CGAN import Generator, Discriminator
from utils.slate_formation import get_data_loaders

import torch
import os
import pandas as pd

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
#
# def gradient_penalty():
#     # Calculate Gradient policy
#     epsilon = torch.rand(real_slates.shape[0], 1)
#     epsilon = epsilon.expand(real_slates.size()).to(self.device)
#
#     interpolation = epsilon * real_slates + ((1 - epsilon) * fake_slates)
#     interpolation = torch.autograd.Variable(interpolation, requires_grad=True).to(self.device)
#
#     dis_interpolates, _ = self.discriminator(interpolation, user_interactions_with_padding,
#                                              number_of_interactions_per_user, response_vector)
#     grad_outputs = torch.ones(dis_interpolates.size()).to(self.device)
#
#     gradients = torch.autograd.grad(outputs=dis_interpolates,
#                                     inputs=interpolation,
#                                     grad_outputs=grad_outputs,
#                                     create_graph=True,
#                                     retain_graph=True,
#                                     only_inputs=True)[0]
#
#     gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA
#
#     return gradients

class FullyConnectedGANExperimentBuilder(ExperimentBuilderGAN):
    CRITIC_ITERS = 5
    LAMBDA = 10
    weight_clip = 0.1

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
        user_interactions_with_padding = values_to_unpack[1].to(self.device)
        number_of_interactions_per_user = values_to_unpack[2].to(self.device)
        response_vector = torch.full((user_interactions_with_padding.shape[0], self.configs['slate_size']), 1,
                                     device=self.device, dtype=torch.float32)

        noise = torch.randn(user_interactions_with_padding.shape[0], self.configs['noise_hidden_dims'],
                            dtype=torch.float32, device=self.device)

        fake_slates = self.generator(user_interactions_with_padding, number_of_interactions_per_user, response_vector, noise,
                                     inference=True)

        return fake_slates

    def update_discriminator(self, real_slates, user_interactions_with_padding, number_of_interactions_per_user, response_vector):
        self.optimizer_dis.zero_grad()

        dis_real, _ = self.discriminator(real_slates, user_interactions_with_padding, number_of_interactions_per_user, response_vector)
        dis_loss_real = dis_real.mean()

        # Generate fake slates
        noise = torch.randn(user_interactions_with_padding.shape[0], self.configs['noise_hidden_dims'],
                            dtype=torch.float32, device=self.device)

        fake_slates = self.generator(user_interactions_with_padding, number_of_interactions_per_user, response_vector,
                                     noise)

        dis_fake, _ = self.discriminator(fake_slates.detach(), user_interactions_with_padding,
                                         number_of_interactions_per_user, response_vector)
        dis_loss_fake = dis_fake.mean()

        d_loss = dis_loss_fake - dis_loss_real
        d_loss.backward()

        self.optimizer_dis.step()

        # Clip weights of discriminator
        for p in self.discriminator.parameters():
            p.data.clamp_(-self.weight_clip, self.weight_clip)

        return d_loss.item()

    def update_generator(self, real_slates, user_interactions_with_padding, number_of_interactions_per_user, response_vector):
        self.optimizer_gen.zero_grad()

        noise = torch.randn(user_interactions_with_padding.shape[0], self.configs['noise_hidden_dims'],
                            dtype=torch.float32, device=self.device)

        fake_slates = self.generator(user_interactions_with_padding, number_of_interactions_per_user, response_vector, noise)
        fake_loss, phi_fake = self.discriminator(fake_slates, user_interactions_with_padding, number_of_interactions_per_user, response_vector)
        g_loss = -1 * fake_loss.mean()
        g_loss.backward(retain_graph=True)

        _, phi_real = self.discriminator(real_slates, user_interactions_with_padding, number_of_interactions_per_user, response_vector)

        gdpp_loss = GDPPLoss(phi_real, phi_fake)

        self.optimizer_gen.step()

        return float(g_loss) + float(gdpp_loss)


def experiments_run():
    configs = extract_args_from_json()
    print(configs)
    set_seeds(configs['seed'])

    train_loader, test_loader, data_configs = get_data_loaders(configs, True)

    print('number of movies: ', train_loader.dataset.number_of_movies)

    response_vector_dims = 1

    generator = Generator(train_loader.dataset.number_of_movies, configs['slate_size'], configs['embed_dims'],
                          configs['noise_hidden_dims'], configs['hidden_layers_dims_gen'], response_vector_dims,
                          configs['gen_dropout'])

    print(generator)

    discriminator = Discriminator(train_loader.dataset.number_of_movies, configs['slate_size'], configs['embed_dims'],
                                  configs['hidden_layers_dims_dis'], response_vector_dims, configs['dis_dropout'])
    print(discriminator)

    experiment_builder = FullyConnectedGANExperimentBuilder(generator, discriminator, train_loader, test_loader, configs,
                                                            print_learnable_parameters=True)
    experiment_builder.run_experiment()


if __name__ == '__main__':
    experiments_run()
