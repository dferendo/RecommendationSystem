import pandas as pd
import numpy as np

# Configs
DATA_LOCATION = './dataset/ml-latest-small/ratings.csv'
SLATE_SIZE = 6
IMPLICIT_RATING = 4
MINIMUM_MOVIE_INTERACTION = 5
MINIMUM_USER_INTERACTION = 13

# Loading
df_all = pd.read_csv(DATA_LOCATION)

# Make dataset implicit (ie User had interaction/User did not have interaction)
df_all = df_all[df_all['rating'] >= IMPLICIT_RATING]
df_all.loc[df_all['rating'] >= IMPLICIT_RATING, 'rating'] = 1

# Pre-processing
if MINIMUM_MOVIE_INTERACTION != -1:
    movies_interactions_counts = df_all.groupby(['movieId']).count()
    # For each interaction, check whether the movieId occurred more than MINIMUM_MOVIE_INTERACTION times
    df_all = df_all.loc[df_all['movieId'].isin(movies_interactions_counts[movies_interactions_counts['timestamp'] >=
                                                                          MINIMUM_MOVIE_INTERACTION].index)]

if MINIMUM_USER_INTERACTION != -1:
    users_interactions_counts = df_all.groupby(['userId']).count()
    # For each interaction, check whether the userId occurred more than MINIMUM_MOVIE_INTERACTION times
    df_all = df_all.loc[df_all['userId'].isin(users_interactions_counts[users_interactions_counts['timestamp'] >=
                                                                        MINIMUM_USER_INTERACTION].index)]

# Sorting is done so that the model does not have access to future interactions
df_sorted_by_timestamp = df_all.sort_values(by=['timestamp'])
df_interactions_sorted = df_sorted_by_timestamp.groupby(['userId'])['movieId'].apply(list)

print(len(df_sorted_by_timestamp))

# test_interactions = []
# train_interactions = []
#
# # Splitting the dataset by time-step instead of random
# for user_id, user_interactions in df_interactions_sorted.items():
#     # Remove users without X interactions
#     if len(user_interactions) < SLATE_SIZE:
#         continue
#
#     user_interaction_train = user_interactions[:-SLATE_SIZE]
#     user_interaction_test = user_interactions[-SLATE_SIZE:]
#
#     train_interactions.append((user_id, user_interaction_train))
#     test_interactions.append((user_id, user_interaction_test))
#
# print(train_interactions)
# print(test_interactions)