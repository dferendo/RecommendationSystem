import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.dim = dim

        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(self.dim, self.dim, 5, padding=2),
            nn.ReLU(True),
            nn.Conv1d(self.dim, self.dim, 5, padding=2),
        )

    def forward(self, input):
        output = self.res_block(input)
        return input + (0.3 * output)


class Generator(nn.Module):
    def __init__(self, num_of_movies, sequence_len, response_dims, embed_dims, noise_dim, dim, batch_size):
        super(Generator, self).__init__()
        self.dim = dim
        self.sequence_len = sequence_len
        self.num_of_movies = num_of_movies
        self.batch_size = batch_size
        self.response_dims = response_dims
        self.embed_dims = embed_dims

        # Index work from 0 - (num_of_movies - 1). Thus, we use num_of_movies as a padding index
        self.padding_idx = self.num_of_movies

        # +1 for the padding index
        self.embedding_movies = torch.nn.Embedding(num_embeddings=self.num_of_movies + 1,
                                                   embedding_dim=self.embed_dims,
                                                   padding_idx=self.padding_idx)

        input_dims = noise_dim + self.embed_dims + self.response_dims

        self.fc1 = nn.Linear(input_dims, self.dim * self.sequence_len)
        self.block = nn.Sequential(
            ResBlock(self.dim),
            ResBlock(self.dim),
            ResBlock(self.dim),
            ResBlock(self.dim),
            ResBlock(self.dim),
        )

        self.conv1 = nn.Conv1d(self.dim, self.num_of_movies, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, user_interactions_with_padding, number_of_interactions_per_user, response_vector, noise,
                inference=False):
        movie_embedding = self.embedding_movies(user_interactions_with_padding)
        user_embedding = torch.sum(movie_embedding, dim=1) / number_of_interactions_per_user.unsqueeze(dim=1)
        response_vector = response_vector.sum(dim=1).unsqueeze(dim=1)

        gen_input = torch.cat((noise, user_embedding, response_vector), dim=1)

        output = self.fc1(gen_input)
        output = output.view(-1, self.dim, self.sequence_len) # (BATCH_SIZE, DIM, SEQ_LEN)
        output = self.block(output)
        output = self.conv1(output)
        output = output.transpose(1, 2)
        shape = output.size()
        output = output.contiguous()
        output = output.view(self.batch_size*self.sequence_len, -1)
        output = self.softmax(output)
        output = output.view(shape) # (BATCH_SIZE, SEQ_LEN, len(charmap))

        if inference:
            return torch.argmax(output, dim=2)

        return output

    def reset_parameters(self):
        """
        TODO
        Re-initializes the networks parameters
        """
        pass


class Discriminator(nn.Module):
    def __init__(self, dim, sequence_len, num_of_movies, embed_dims, response_dims):
        super(Discriminator, self).__init__()
        self.dim = dim
        self.sequence_len = sequence_len
        self.num_of_movies = num_of_movies
        self.embed_dims = embed_dims
        self.response_dims = response_dims

        self.block = nn.Sequential(
            ResBlock(self.dim),
            ResBlock(self.dim),
            ResBlock(self.dim),
            ResBlock(self.dim),
            ResBlock(self.dim),
        )
        self.conv1d = nn.Conv1d(num_of_movies, self.dim, 1)

        # Index work from 0 - (num_of_movies - 1). Thus, we use num_of_movies as a padding index
        self.padding_idx = self.num_of_movies

        # +1 for the padding index
        self.embedding_movies = torch.nn.Embedding(num_embeddings=self.num_of_movies + 1,
                                                   embedding_dim=self.embed_dims,
                                                   padding_idx=self.padding_idx)

        linear_input = (self.sequence_len * self.dim) + self.embed_dims + self.response_dims

        self.linear = nn.Linear(linear_input, 1)

    def forward(self, slate_input, user_interactions_with_padding, number_of_interactions_per_user,
                response_vector):
        output = slate_input.transpose(1, 2)  # (BATCH_SIZE, len(charmap), SEQ_LEN)

        output = self.conv1d(output)
        output = self.block(output)
        output = output.view(-1, self.sequence_len * self.dim)

        movie_embedding = self.embedding_movies(user_interactions_with_padding)
        user_embedding = torch.sum(movie_embedding, dim=1) / number_of_interactions_per_user.unsqueeze(dim=1)
        response_vector = response_vector.sum(dim=1).unsqueeze(dim=1)

        output = torch.cat((output, user_embedding, response_vector), dim=1)

        output = self.linear(output)

        return output, None

    def reset_parameters(self):
        """
        TODO
        Re-initializes the networks parameters
        """
        pass
