import pandas as pd
import numpy as np
import tqdm
import sys
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from models.ListCVAE import ListCVAE
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
        return self.training_user[idx], self.interactions[idx], np.array(self.number_of_interactions[idx]), np.array(self.training_slate[idx]), np.ones(6)


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
        return np.array(self.optimum_slates[idx]), self.interactions[idx], np.array(self.number_of_interactions[idx]), np.ones(6)


train_dataset = TrainingDataset(training_user, training_interactions, training_slate, len(all_movies_indexes))
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=4, drop_last=True)

test_dataset = TestingDataset(optimum_slate, testing_interactions, train_dataset.max_training_interactions, len(all_movies_indexes))
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=4, drop_last=True)

device = torch.device("cuda")

model = ListCVAE(len(all_movies_indexes), 6, 1, 16, [20, 40], 16, [20, 40], device)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)
criterion = torch.nn.CrossEntropyLoss()


def loss_function(recon_slates, slates, prior_mu, prior_log_variance):
    recon_slates = recon_slates.view(recon_slates.shape[0] * recon_slates.shape[1], recon_slates.shape[2])
    slates = slates.view(slates.shape[0] * slates.shape[1])

    entropy_loss = criterion(recon_slates, slates)
    KLD = -0.5 * torch.sum(1 + prior_log_variance - prior_mu.pow(2) - prior_log_variance.exp())
    return entropy_loss + KLD


for epoch in range(0, 100):
    with tqdm.tqdm(total=len(train_loader), file=sys.stdout) as pbar:
        for values in train_loader:
            optimizer.zero_grad()

            users = values[0].to(device)
            interactions = values[1].to(device)
            number_of_interactions = values[2].to(device)
            slates = values[3].to(device)
            response = values[4].to(device).float()

            decoder_out, mu, log_variance = model(slates, interactions, number_of_interactions, response)
            loss = loss_function(decoder_out, slates, mu, log_variance)

            loss.backward()
            optimizer.step()

            pbar.update(1)
            pbar.set_description(f"loss: {float(loss):.4f}")

    predicted_slates = []
    truth_slates = []

    for values in test_loader:
        optimum_slates = values[0]
        interactions = values[1].to(device)
        number_of_interactions = values[2].to(device)
        response = values[3].to(device).float()

        slates = model.inference(interactions, number_of_interactions, response)

        predicted_slates.append(slates)
        truth_slates.append(optimum_slates)

    predicted_slates = torch.cat(predicted_slates, dim=0)
    truth_slates = torch.cat(truth_slates, dim=0)

    diversity = movie_diversity(predicted_slates, len(all_movies_indexes))

    predicted_slates = predicted_slates.cpu()
    precision, hr = precision_hit_ratio(predicted_slates, truth_slates)

    print(f'precision {precision}, hr {hr}, diversity {diversity}')
