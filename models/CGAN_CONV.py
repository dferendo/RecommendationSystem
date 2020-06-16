import torch.nn as nn
import torch


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

        return input + (0.3*output)


class Generator(nn.Module):
    def __init__(self, number_of_movies, sequence_len, noise_dim, conv_dim):
        super(Generator, self).__init__()
        self.sequence_len = sequence_len
        self.noise_dim = noise_dim
        self.conv_dim = conv_dim
        self.number_of_movies = number_of_movies

        self.fc1 = nn.Linear(noise_dim, conv_dim * sequence_len)
        self.block = nn.Sequential(
            ResBlock(self.conv_dim),
            ResBlock(self.conv_dim),
            ResBlock(self.conv_dim),
            ResBlock(self.conv_dim),
            ResBlock(self.conv_dim),
        )

        self.conv1 = nn.Conv1d(conv_dim, self.number_of_movies, 1)
        self.softmax = nn.Softmax()

    def forward(self, noise):
        output = self.fc1(noise)
        output = output.view(-1, self.conv_dim, self.sequence_len)
        output = self.block(output)
        output = self.conv1(output)
        output = output.transpose(1, 2)
        shape = output.size()
        output = output.contiguous()
        output = output.view(noise.shape[0] * self.sequence_len, -1)
        output = self.softmax(output)

        return output.view(shape)

    def reset_parameters(self):
        """
        TODO
        Re-initializes the networks parameters
        """
        pass


class Discriminator(nn.Module):
    def __init__(self, number_of_movies, sequence_len, conv_dim):
        super(Discriminator, self).__init__()
        self.conv_dim = conv_dim
        self.number_of_movies = number_of_movies
        self.sequence_len = sequence_len

        self.block = nn.Sequential(
            ResBlock(self.conv_dim),
            ResBlock(self.conv_dim),
            ResBlock(self.conv_dim),
            ResBlock(self.conv_dim),
            ResBlock(self.conv_dim),
        )
        self.conv1d = nn.Conv1d(self.number_of_movies, self.conv_dim, 1)
        self.linear = nn.Linear(self.sequence_len * self.conv_dim, 1)

    def forward(self, input):
        output = input.transpose(1, 2)
        output = self.conv1d(output)
        output = self.block(output)
        output = output.view(-1, self.sequence_len * self.conv_dim)
        output = self.linear(output)

        return output

    def reset_parameters(self):
        """
        TODO
        Re-initializes the networks parameters
        """
        pass