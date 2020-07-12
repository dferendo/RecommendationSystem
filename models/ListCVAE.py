import torch
import torch.utils.data
from torch import nn
import numpy as np


class Parameters:
    def __init__(self, batch_norm, dropout, activation_function):
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.activation_function = activation_function


class ListCVAE(nn.Module):
    def __init__(self, num_of_movies, slate_size, response_dims, embed_dims, encoder_dims, latent_dims, decoder_dims,
                 prior_dims, device, encoder_params, decoder_params, prior_params, gdpp_active):
        super(ListCVAE, self).__init__()
        self.device = device

        self.num_of_movies = num_of_movies
        self.slate_size = slate_size
        self.response_dims = response_dims
        self.embed_dims = embed_dims
        self.encoder_dims = encoder_dims
        self.latent_dims = latent_dims
        self.gdpp_active = gdpp_active

        # Index work from 0 - (num_of_movies - 1). Thus, we use num_of_movies as a padding index
        self.padding_idx = self.num_of_movies

        # +1 for the padding index
        self.embedding_movies = torch.nn.Embedding(num_embeddings=self.num_of_movies + 1,
                                                   embedding_dim=self.embed_dims,
                                                   padding_idx=self.padding_idx)

        # Encoder
        layers_block = []
        input_dims = (self.embed_dims * self.slate_size) + self.embed_dims + self.response_dims

        for out_dims in encoder_dims:
            layers_block.extend(self.encoder_block(input_dims, out_dims, encoder_params))
            input_dims = out_dims

        self.encoder_layers = nn.Sequential(
            *layers_block
        )

        self.encoder_mu = nn.Linear(input_dims, self.latent_dims)
        self.encoder_log_variance = nn.Linear(input_dims, self.latent_dims)

        # Decoder
        layers_block = []
        input_dims = self.latent_dims + self.embed_dims + self.response_dims

        for out_dims in decoder_dims:
            layers_block.extend(self.decoder_block(input_dims, out_dims, decoder_params))
            input_dims = out_dims

        self.decoder_layers = nn.Sequential(
            *layers_block,
            nn.Linear(input_dims, self.embed_dims * self.slate_size)
        )

        # Prior
        layers_block = []
        input_dims = self.embed_dims + self.response_dims

        for out_dims in prior_dims:
            layers_block.extend(self.prior_block(input_dims, out_dims, prior_params))
            input_dims = out_dims

        self.prior_layers = nn.Sequential(
            *layers_block
        )

        self.prior_mu = nn.Linear(input_dims, self.latent_dims)
        self.prior_log_variance = nn.Linear(input_dims, self.latent_dims)

    @staticmethod
    def encoder_block(in_feat, out_feat, parameters):
        block = [nn.Linear(in_feat, out_feat)]

        if parameters.batch_norm:
            block.append(nn.BatchNorm1d(out_feat))

        if parameters.activation_function == 'leaky':
            block.append(nn.LeakyReLU())
        elif parameters.activation_function == 'relu':
            block.append(nn.ReLU())
        else:
            block.append(nn.Tanh())

        if parameters.dropout:
            block.append(nn.Dropout(parameters.dropout))

        return block

    @staticmethod
    def decoder_block(in_feat, out_feat, parameters):
        block = [nn.Linear(in_feat, out_feat)]

        if parameters.batch_norm:
            block.append(nn.BatchNorm1d(out_feat))

        if parameters.activation_function == 'leaky':
            block.append(nn.LeakyReLU())
        elif parameters.activation_function == 'relu':
            block.append(nn.ReLU())
        else:
            block.append(nn.Tanh())

        if parameters.dropout:
            block.append(nn.Dropout(parameters.dropout))

        return block

    @staticmethod
    def prior_block(in_feat, out_feat, parameters):
        block = [nn.Linear(in_feat, out_feat)]

        if parameters.batch_norm:
            block.append(nn.BatchNorm1d(out_feat))

        if parameters.activation_function == 'leaky':
            block.append(nn.LeakyReLU())
        elif parameters.activation_function == 'relu':
            block.append(nn.ReLU())
        else:
            block.append(nn.Tanh())

        if parameters.dropout:
            block.append(nn.Dropout(parameters.dropout))

        return block

    def encode(self, slate_inputs, conditioned_info):
        # Slate Embeds
        slate_embeds = self.embedding_movies(slate_inputs)
        slate_embeds = slate_embeds.view(slate_embeds.shape[0], self.slate_size * self.embed_dims)

        encoder_input = torch.cat((slate_embeds, conditioned_info), dim=1)

        out = self.encoder_layers(encoder_input)

        last_hidden = out

        mu = self.encoder_mu(last_hidden)
        log_variance = self.encoder_log_variance(last_hidden)

        return mu, log_variance, last_hidden

    def reparameterize(self, mu, log_variance):
        """
        https://stats.stackexchange.com/questions/199605/how-does-the-reparameterization-trick-for-vaes-work-and-why-is-it-important
        :param mu:
        :param log_variance:
        :return:
        """
        std = torch.exp(0.5 * log_variance)
        eps = torch.rand_like(std, device=self.device)

        return mu + eps * std

    def decode(self, z, conditioned_info):
        decoder_input = torch.cat((z, conditioned_info), dim=1)

        out = self.decoder_layers(decoder_input)

        all_movies = torch.arange(self.num_of_movies, device=self.device)
        all_movies_embedding = self.embedding_movies(all_movies).T

        out = out.view(out.shape[0], self.slate_size, self.embed_dims)

        out = torch.matmul(out, all_movies_embedding)

        return out

    def forward(self, slate_inputs, user_interactions_with_padding, number_of_interactions_per_user, response_vector):
        last_hidden_fake = None

        # Personalized
        movie_embedding = self.embedding_movies(user_interactions_with_padding)
        user_embedding = torch.sum(movie_embedding, dim=1) / number_of_interactions_per_user.unsqueeze(dim=1)
        response_vector = response_vector.sum(dim=1).unsqueeze(dim=1)

        conditioned_info = torch.cat((user_embedding, response_vector), dim=1)

        # Encoder
        mu, log_variance, last_hidden_real = self.encode(slate_inputs, conditioned_info)

        # Decoder
        z = self.reparameterize(mu, log_variance)
        decoder_out = self.decode(z, conditioned_info)

        # Prior
        prior_out = self.prior_layers(conditioned_info)
        prior_mu = self.prior_mu(prior_out)
        prior_log_variance = self.prior_log_variance(prior_out)

        if self.gdpp_active:
            fake_slates = self.get_slates(decoder_out)
            _, _, last_hidden_fake = self.encode(fake_slates, conditioned_info)

        return decoder_out, mu, log_variance, prior_mu, prior_log_variance, last_hidden_real, last_hidden_fake

    def get_slates(self, decoder_out):
        slates = []
        masking = torch.zeros([decoder_out.shape[0], decoder_out.shape[2]], device=self.device, dtype=torch.float32)

        for slate_item in range(self.slate_size):
            slate_output = decoder_out[:, slate_item, :]
            slate_output = slate_output + masking
            slate_item = torch.argmax(slate_output, dim=1)

            slates.append(slate_item)
            masking = masking.scatter_(1, slate_item.unsqueeze(dim=1), float('-inf'))

        return torch.stack(slates, dim=1)

    def inference(self, user_interactions_with_padding, number_of_interactions_per_user, response_vector):
        # Personalized
        movie_embedding = self.embedding_movies(user_interactions_with_padding)
        user_embedding = torch.sum(movie_embedding, dim=1) / number_of_interactions_per_user.unsqueeze(dim=1)
        response_vector = response_vector.sum(dim=1).unsqueeze(dim=1)

        conditioned_info = torch.cat((user_embedding, response_vector), dim=1)

        # Prior
        prior_out = self.prior_layers(conditioned_info)
        prior_mu = self.prior_mu(prior_out)
        prior_log_variance = self.prior_log_variance(prior_out)

        z = self.reparameterize(prior_mu, prior_log_variance)

        decoder_out = self.decode(z, conditioned_info)

        return self.get_slates(decoder_out)

    def reset_parameters(self):
        """
        TODO
        Re-initializes the networks parameters
        """
        pass
