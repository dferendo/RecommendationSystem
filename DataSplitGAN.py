import pandas as pd
import numpy as np
import tqdm
import sys
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from models.CGAN import Generator, Discriminator
import torch
from utils.evaluation_metrics import precision_hit_ratio, movie_diversity

# Get data
DATA_LOCATION = './dataset/ml-1m'
IMPLICIT_RATING = 0
TEST_SET_SIZE = 0.15
MINIMUM_USER_INTERACTION = 10

df_all = pd.read_csv(f'{DATA_LOCATION}/ratings.csv')

# Make dataset implicit (ie User had interaction/User did not have interaction)
df_all = df_all[df_all['rating'] >= IMPLICIT_RATING]
df_all.loc[df_all['rating'] >= IMPLICIT_RATING, 'rating'] = 1

df_sorted_by_timestamp = df_all.sort_values(by=['timestamp'])
test_size = int(len(df_sorted_by_timestamp) * TEST_SET_SIZE)
test_indexes_start = len(df_sorted_by_timestamp) - test_size

df_train, df_test = np.split(df_sorted_by_timestamp, [test_indexes_start])

if MINIMUM_USER_INTERACTION != -1:
    users_interactions_counts = df_train.groupby(['userId']).count()
    # For each interaction, check whether the userId occurred more than MINIMUM_USER_INTERACTION times
    df_train = df_train.loc[df_train['userId'].isin(users_interactions_counts[users_interactions_counts['timestamp'] >=
                                                                              MINIMUM_USER_INTERACTION].index)]

# Remove any users that do not appear in the training set
df_test = df_test.loc[df_test['userId'].isin(df_train['userId'].unique())]

# Remove any movies that do not appear in the training set from the test set
df_test = df_test.loc[df_test['movieId'].isin(df_train['movieId'].unique())]


all_users = df_train['userId'].unique()
all_movies = df_train['movieId'].unique()

all_users_indexes = np.arange(len(all_users))
all_movies_indexes = np.arange(len(all_movies))

print(f'users: {all_users_indexes}')
print(f'movies: {all_movies_indexes}')

df_train['userId'] = df_train['userId'].replace(all_users, all_users_indexes)
df_train['movieId'] = df_train['movieId'].replace(all_movies, all_movies_indexes)

df_test['userId'] = df_test['userId'].replace(all_users, all_users_indexes)
df_test['movieId'] = df_test['movieId'].replace(all_movies, all_movies_indexes)

df_train.to_csv(f'{DATA_LOCATION}/preprocessed/train.csv', index=False)
df_test.to_csv(f'{DATA_LOCATION}/preprocessed/test.csv', index=False)

with open(f'{DATA_LOCATION}/preprocessed/user_id_to_index.csv', 'w') as f:
    for user_id, user_index in zip(all_users, all_users_indexes):
        f.write(f"{user_id},{user_index}\n")

with open(f'{DATA_LOCATION}/preprocessed/movie_id_to_index.csv', 'w') as f:
    for movie_id, movie_index in zip(all_movies, all_movies_indexes):
        f.write(f"{movie_id},{movie_index}\n")


training_grouped_users = df_train.groupby(['userId'])['movieId'].apply(list)

training_user = []
training_interactions = []
training_slate = []

for user, user_interactions in training_grouped_users.items():
    # Get the possible index of movieIds that we can sample for this user
    # movies_with_no_interactions_with_user = np.setxor1d(all_movies_that_can_be_sampled, user_interactions)
    training_user.append(user)
    training_interactions.append(user_interactions[:-6])
    training_slate.append(user_interactions[-6:])


class TrainingDataset(Dataset):
    def __init__(self, training_user, training_interactions, training_slate, padding_idx):
        self.training_user = training_user
        self.training_slate = training_slate
        self.padding_idx = padding_idx

        self.max_training_interactions = len(max(training_interactions, key=len))

        self.interactions = []
        self.number_of_interactions = []

        for interaction in training_interactions:
            padded_interactions = np.full(self.max_training_interactions, padding_idx)
            padded_interactions[0:len(interaction)] = interaction

            self.number_of_interactions.append(len(interaction))
            self.interactions.append(padded_interactions)

    def __len__(self):
        return len(self.training_user)

    def __getitem__(self, idx):
        a = np.array(self.training_slate[idx])
        one_hot_slates = np.zeros((a.size, self.padding_idx))
        one_hot_slates[np.arange(a.size), a] = 1

        one_hot_slates = one_hot_slates.reshape(a.size * self.padding_idx)

        return self.training_user[idx], self.interactions[idx], np.array(self.number_of_interactions[idx]), one_hot_slates, np.ones(6)


# Evaluation
training_grouped_users = df_train.groupby(['userId'])
grouped_users = df_test.groupby(['userId'])['movieId'].apply(list)

TARGET_SIZE = 10
testing_interactions = []
optimum_slate = []

for user, user_interactions in grouped_users.items():
    # Get the possible index of movieIds that we can sample for this user
    # movies_with_no_interactions_with_user = np.setxor1d(all_movies_that_can_be_sampled, user_interactions)
    # training_user.append(user)
    # training_interactions.append(user_interactions[:-6])
    # training_slate.append(user_interactions[-6:])
    if len(user_interactions) >= TARGET_SIZE:
        training_interaction = training_grouped_users.get_group(user)['movieId'].values

        optimum_slate.append(user_interactions[:TARGET_SIZE])
        testing_interactions.append(training_interaction)


class TestingDataset(Dataset):
    def __init__(self, optimum_slates, testing_interactions, longest_interaction, padding_idx):
        self.optimum_slates = optimum_slates
        self.testing_interactions = testing_interactions
        self.longest_interaction = longest_interaction
        self.padding_idx = padding_idx

        self.interactions = []
        self.number_of_interactions = []

        for interaction in training_interactions:
            padded_interactions = np.full(self.longest_interaction, padding_idx)
            padded_interactions[0:len(interaction)] = interaction

            self.number_of_interactions.append(len(interaction))
            self.interactions.append(padded_interactions)

    def __len__(self):
        return len(self.optimum_slates)

    def __getitem__(self, idx):
        a = np.array(self.optimum_slates[idx])
        one_hot_slates = np.zeros((a.size, self.padding_idx))
        one_hot_slates[np.arange(a.size), a] = 1

        one_hot_slates = one_hot_slates.reshape(a.size * self.padding_idx)

        return one_hot_slates, self.interactions[idx], np.array(self.number_of_interactions[idx]), np.ones(6)


train_dataset = TrainingDataset(training_user, training_interactions, training_slate, len(all_movies_indexes))
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=4, drop_last=True)

test_dataset = TestingDataset(optimum_slate, testing_interactions, train_dataset.max_training_interactions, len(all_movies_indexes))
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=4, drop_last=True)

device = torch.device("cuda")

generator = Generator(len(all_movies_indexes), 6, 32, 32, [32, 64], 1)
discriminator = Discriminator(len(all_movies_indexes), 6, 32, [32, 64], 1)

print(generator)
print(discriminator)

generator.to(device)
discriminator.to(device)

optimizer_gen = torch.optim.Adam(generator.parameters(), lr=0.0001, weight_decay=0)
optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=0.0001, weight_decay=0)


def update_discriminator(real_slates, user_interactions_with_padding, number_of_interactions_per_user, response_vector):
    optimizer_dis.zero_grad()

    dis_real, _ = discriminator(real_slates, user_interactions_with_padding, number_of_interactions_per_user,
                                     response_vector)

    dis_real = dis_real.mean()

    # Generate fake slates
    noise = torch.randn(user_interactions_with_padding.shape[0], 32,
                        dtype=torch.float32, device=device)

    fake_slates = generator(user_interactions_with_padding, number_of_interactions_per_user, response_vector,
                                 noise)
    dis_fake, _ = discriminator(fake_slates.detach(), user_interactions_with_padding,
                                     number_of_interactions_per_user, response_vector)
    dis_fake = dis_fake.mean()

    # Calculate Gradient policy
    epsilon = torch.rand(real_slates.shape[0], 1)
    epsilon = epsilon.expand(real_slates.size()).to(device)

    interpolation = epsilon * real_slates + ((1 - epsilon) * fake_slates)
    interpolation = torch.autograd.Variable(interpolation, requires_grad=True).to(device)

    dis_interpolates, _ = discriminator(interpolation, user_interactions_with_padding,
                                             number_of_interactions_per_user, response_vector)
    grad_outputs = torch.ones(dis_interpolates.size()).to(device)

    gradients = torch.autograd.grad(outputs=dis_interpolates,
                                    inputs=interpolation,
                                    grad_outputs=grad_outputs,
                                    create_graph=True,
                                    retain_graph=True,
                                    only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10

    d_loss = dis_real - dis_fake + gradient_penalty
    d_loss.backward()

    optimizer_dis.step()

    return d_loss


def update_generator(real_slates, user_interactions_with_padding, number_of_interactions_per_user, response_vector):
    optimizer_gen.zero_grad()
    optimizer_dis.zero_grad()

    noise = torch.randn(user_interactions_with_padding.shape[0], 32,
                        dtype=torch.float32, device=device)

    fake_slates = generator(user_interactions_with_padding, number_of_interactions_per_user, response_vector, noise)
    fake_loss, fake_h = discriminator(fake_slates, user_interactions_with_padding, number_of_interactions_per_user, response_vector)
    fake_loss = fake_loss.mean()
    fake_loss.backward(retain_graph=True)

    _, real_h = discriminator(real_slates, user_interactions_with_padding, number_of_interactions_per_user, response_vector)
    # gdpp_loss = GDPPLoss(real_h, fake_h, backward=True)

    # g_loss = -fake_loss + gdpp_loss
    g_loss = -fake_loss
    optimizer_gen.step()
    return g_loss


for epoch in range(0, 100):
    with tqdm.tqdm(total=len(train_loader), file=sys.stdout) as pbar:
        for idx, values in enumerate(train_loader):
            optimizer_dis.zero_grad()

            users = values[0].to(device)
            interactions = values[1].to(device)
            number_of_interactions = values[2].to(device)
            slates = values[3].to(device).float()
            response = values[4].to(device).float()

            loss_diss = update_discriminator(slates, interactions, number_of_interactions, response)

            if idx != 0 and idx % 5 == 0:
                for p in discriminator.parameters():
                    p.requires_grad = False

                loss_gen = update_generator(slates, interactions, number_of_interactions, response)

                for p in discriminator.parameters():
                    p.requires_grad = True
            else:
                loss_gen = None

            pbar.update(1)
            pbar.set_description(f"loss: {float(loss_diss):.4f}")

    predicted_slates = []
    truth_slates = []

    for values in test_loader:
        optimum_slates = values[0]
        interactions = values[1].to(device)
        number_of_interactions = values[2].to(device)
        response = values[3].to(device).float()

        response_vector = torch.full((10, 6), 1,  device=device, dtype=torch.float32)

        noise = torch.randn(interactions.shape[0], 32, dtype=torch.float32, device=device)

        fake_slates = generator(interactions, number_of_interactions, response_vector, noise,
                                     inference=True)

        predicted_slates.append(fake_slates)
        truth_slates.append(optimum_slates)

    predicted_slates = torch.cat(predicted_slates, dim=0)
    truth_slates = torch.cat(truth_slates, dim=0)

    diversity = movie_diversity(predicted_slates, len(all_movies_indexes))

    predicted_slates = predicted_slates.cpu()
    precision, hr = precision_hit_ratio(predicted_slates, truth_slates)

    print(f'precision {precision}, hr {hr}, diversity {diversity}')
