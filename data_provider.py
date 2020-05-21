import pandas as pd
import numpy as np

DATA_LOCATION = './dataset/ml-latest-small/ratings.csv'
SLATE_SIZE = 6

df_all = pd.read_csv(DATA_LOCATION)

# Make dataset implicit (ie User had interaction/User did not have interaction)
df_all = df_all[df_all['rating'] >= 4]
df_all.loc[df_all['rating'] >= 4, 'rating'] = 1

df_sorted_by_timestamp = df_all.sort_values(by=['timestamp'])
df_interactions_sorted = df_sorted_by_timestamp.groupby(['userId'])['movieId'].apply(list)

test_interactions = []
train_interactions = []

for user_id, user_interactions in df_interactions_sorted.items():
    if len(user_interactions) < SLATE_SIZE:
        continue

    train_interactions = user_interactions[:-SLATE_SIZE]
    test_interactions = user_interactions[-SLATE_SIZE:]

    test_interactions.append((user_id, test_interactions))
    train_interactions.append((user_id, train_interactions))

print(test_interactions)
print(train_interactions)
