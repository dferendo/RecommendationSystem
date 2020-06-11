import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, num_of_movies, slate_size, embed_dims, noise_dim, hidden_layers_dims):
        super(Generator, self).__init__()
        self.num_of_movies = num_of_movies
        self.embed_dims = embed_dims
        self.noise_dim = noise_dim
        self.slate_size = slate_size

        # Index work from 0 - (num_of_movies - 1). Thus, we use num_of_movies as a padding index
        self.padding_idx = self.num_of_movies

        # +1 for the padding index
        self.embedding_movies = torch.nn.Embedding(num_embeddings=self.num_of_movies + 1,
                                                   embedding_dim=self.embed_dims,
                                                   padding_idx=self.padding_idx)

        layers_block = []
        in_dims = self.embed_dims + self.noise_dim

        for out_dims in hidden_layers_dims:
            layers_block.extend(self.gen_block(in_dims, out_dims))
            in_dims = out_dims

        self.model_layers = nn.Sequential(
            *layers_block
        )

        # Will be used to hold the output linear layers
        self.output_dict = nn.ModuleDict()
        self.output_dict['output_tanh'] = nn.Tanh()
        self.output_dict['softmax'] = nn.Softmax(dim=1)

        for i in range(self.slate_size):
            self.output_dict[f'output_linear{i}'] = nn.Linear(in_dims, self.num_of_movies)

    def forward(self, user_interactions_with_padding, number_of_interactions_per_user, noise, inference=False):
        movie_embedding = self.embedding_movies(user_interactions_with_padding)
        user_embedding = torch.sum(movie_embedding, dim=1) / number_of_interactions_per_user.unsqueeze(dim=1)

        gen_input = torch.cat((noise, user_embedding), dim=-1)
        out = self.model_layers(gen_input)

        slate_outputs = []

        for i in range(self.slate_size):
            slate_embed_output = self.output_dict[f'output_linear{i}'](out)
            slate_embed_output = self.output_dict['output_tanh'](slate_embed_output)

            slate_embed_output = self.output_dict['softmax'](slate_embed_output)

            if inference:
                slate_embed_output = torch.argmax(slate_embed_output, dim=1).unsqueeze(dim=1)

            slate_outputs.append(slate_embed_output)

        slates = torch.cat(slate_outputs, dim=1)

        return slates

    @staticmethod
    def gen_block(in_feat, out_feat, normalize=True, dropout=0.2):
        layers = [nn.Linear(in_feat, out_feat)]

        if normalize:
            layers.append(nn.BatchNorm1d(num_features=out_feat))
        if dropout:
            layers.append(nn.Dropout(dropout))

        layers.append(nn.LeakyReLU(0.01, inplace=True))

        return layers

    def reset_parameters(self):
        """
        TODO
        Re-initializes the networks parameters
        """
        pass


class Discriminator(nn.Module):
    def __init__(self, num_of_movies, slate_size, embed_dims, hidden_layers_dims):
        super(Discriminator, self).__init__()
        self.num_of_movies = num_of_movies
        self.embed_dims = embed_dims

        # Index work from 0 - (num_of_movies - 1). Thus, we use num_of_movies as a padding index
        self.padding_idx = self.num_of_movies

        # +1 for the padding index
        self.embedding_movies = torch.nn.Embedding(num_embeddings=self.num_of_movies + 1,
                                                   embedding_dim=self.embed_dims,
                                                   padding_idx=self.padding_idx)

        layers_block = []

        input_dims = (self.num_of_movies * slate_size) + self.embed_dims

        for out_dims in hidden_layers_dims:
            layers_block.extend(self.dis_block(input_dims, out_dims))
            input_dims = out_dims

        self.model_layers = nn.Sequential(
            *layers_block,
            nn.Linear(input_dims, 1),
            nn.Sigmoid()
        )

    def forward(self, slate_input, user_interactions_with_padding, number_of_interactions_per_user):
        # Concatenate label embedding and image to produce input
        movie_embedding = self.embedding_movies(user_interactions_with_padding)
        user_embedding = torch.sum(movie_embedding, dim=1) / number_of_interactions_per_user.unsqueeze(dim=1)

        dim_input = torch.cat((slate_input, user_embedding), dim=1)

        out = self.model_layers(dim_input)

        return out

    @staticmethod
    def dis_block(in_feat, out_feat, normalize=False, dropout=0.2):
        layers = [nn.Linear(in_feat, out_feat)]

        if normalize:
            layers.append(nn.BatchNorm1d(num_features=out_feat))
        if dropout:
            layers.append(nn.Dropout(dropout))

        layers.append(nn.LeakyReLU(0.01, inplace=True))

        return layers

    def reset_parameters(self):
        """
        TODO
        Re-initializes the networks parameters
        """
        pass
