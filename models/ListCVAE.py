import torch
import torch.utils.data
from torch import nn


class ListCVAE(nn.Module):
    def __init__(self, num_of_movies, slate_size, response_dims, embed_dims, encoder_dims, latent_dims, decoder_dims,
                 prior_dims, device):
        super(ListCVAE, self).__init__()
        self.device = device

        self.num_of_movies = num_of_movies
        self.slate_size = slate_size
        self.response_dims = response_dims
        self.embed_dims = embed_dims
        self.encoder_dims = encoder_dims
        self.latent_dims = latent_dims

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
            layers_block.extend(self.encoder_block(input_dims, out_dims))
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
            layers_block.extend(self.decoder_block(input_dims, out_dims))
            input_dims = out_dims

        self.decoder_layers = nn.Sequential(
            *layers_block,
            nn.Linear(input_dims, self.embed_dims * self.slate_size)
        )

        # Prior
        layers_block = []
        input_dims = self.embed_dims + self.response_dims

        for out_dims in prior_dims:
            layers_block.extend(self.prior_block(input_dims, out_dims))
            input_dims = out_dims

        self.prior_layers = nn.Sequential(
            *layers_block
        )

        self.prior_mu = nn.Linear(input_dims, self.latent_dims)
        self.prior_log_variance = nn.Linear(input_dims, self.latent_dims)

    @staticmethod
    def encoder_block(in_feat, out_feat, dropout=None):
        block = [nn.Linear(in_feat, out_feat)]

        # if dropout:
        #     block.append(nn.Dropout(dropout))

        block.append(nn.Tanh())

        return block

    @staticmethod
    def decoder_block(in_feat, out_feat, dropout=None):
        block = [nn.Linear(in_feat, out_feat)]

        # if dropout:
        #     block.append(nn.Dropout(dropout))

        block.append(nn.Tanh())

        return block

    @staticmethod
    def prior_block(in_feat, out_feat, dropout=None):
        block = [nn.Linear(in_feat, out_feat)]

        # if dropout:
        #     block.append(nn.Dropout(dropout))

        block.append(nn.Tanh())

        return block

    def encode(self, slate_inputs, conditioned_info):
        # Slate Embeds
        slate_embeds = self.embedding_movies(slate_inputs)
        slate_embeds = slate_embeds.view(slate_embeds.shape[0], self.slate_size * self.embed_dims)

        encoder_input = torch.cat((slate_embeds, conditioned_info), dim=1)

        out = self.encoder_layers(encoder_input)

        mu = self.encoder_mu(out)
        log_variance = self.encoder_log_variance(out)

        return mu, log_variance

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
        # Personalized
        movie_embedding = self.embedding_movies(user_interactions_with_padding)
        user_embedding = torch.sum(movie_embedding, dim=1) / number_of_interactions_per_user.unsqueeze(dim=1)
        response_vector = response_vector.sum(dim=1).unsqueeze(dim=1)

        conditioned_info = torch.cat((user_embedding, response_vector), dim=1)

        # Encoder
        mu, log_variance = self.encode(slate_inputs, conditioned_info)

        # Decoder
        z = self.reparameterize(mu, log_variance)
        decoder_out = self.decode(z, conditioned_info)

        # Prior
        prior_out = self.prior_layers(conditioned_info)
        prior_mu = self.prior_mu(prior_out)
        prior_log_variance = self.prior_log_variance(prior_out)

        return decoder_out, mu, log_variance, prior_mu, prior_log_variance

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

        return torch.argmax(decoder_out, dim=2)

    def reset_parameters(self):
        """
        TODO
        Re-initializes the networks parameters
        """
        pass
