from utils.arg_parser import extract_args_from_json
from utils.data_provider import split_dataset, load_movie_categories
from utils.reset_seed import set_seeds
import numpy as np
import os
import time
import pandas as pd
import tqdm
import sys


def generate_slate_formation(row_interactions, user_movie_matrix, slate_size, negative_sampling_for_slates,
                             save_location):
    """
    Return the slates. Each slate has a user_id followed by a slate containing
    *slate_size* movie_ids, *slate_size* response vector (whether the user had an interaction or not) and the user
    interactions. All values are in index form (no ids).
    :param row_interactions: All the interactions between users and movies. Each value contains user_id, movie_id,
    rating and timestamp.
    :param user_movie_matrix: [user_id, movie_id] Sparse DataFrame matrix where user_id are the rows and movie_id
    are the columns. The value of each [user_id, movie_id] is whether there are an interaction.
    :param slate_size: The size of the slate
    :param negative_sampling_for_slates: This is an array where each element indicates how many negative examples
    per slate.
    :param save_location: Where to save the slates.
    """
    print("Generating slate formation.....")
    start = time.process_time()

    all_movies_that_can_be_sampled = np.array(user_movie_matrix.columns)

    grouped_users = row_interactions.groupby(['userId'])['movieId'].apply(list)

    print(grouped_users)

    all_samples = []

    with tqdm.tqdm(total=len(grouped_users), file=sys.stdout) as pbar:
        for user_id, user_interactions in grouped_users.items():
            if len(user_interactions) <= slate_size:
                pbar.update(1)
                continue

            # Get the possible index of movieIds that we can sample for this user
            movies_with_no_interactions_with_user = np.setxor1d(all_movies_that_can_be_sampled, user_interactions)

            for negative_samples_amount in negative_sampling_for_slates:
                assert negative_samples_amount <= slate_size

                slate_movies = []
                response_vector = np.zeros(slate_size, dtype=np.int32)

                positive_samples_amount = slate_size - negative_samples_amount

                # The *or None* will return the whole list when we have 0 positive samples
                all_user_interactions = user_interactions[:-positive_samples_amount or None]

                all_user_interactions_indexes = list(map(lambda movie_id: user_movie_matrix.columns.get_loc(movie_id),
                                                         all_user_interactions))

                if positive_samples_amount != 0:
                    positive_samples = user_interactions[-positive_samples_amount:]

                    response_vector[:positive_samples_amount] = 1

                    # Convert to indices
                    positive_indexes = list(map(lambda movie_id: user_movie_matrix.columns.get_loc(movie_id),
                                                positive_samples))

                    slate_movies.extend(positive_indexes)

                if negative_samples_amount != 0:
                    negative_samples = np.random.choice(movies_with_no_interactions_with_user,
                                                        size=negative_samples_amount)

                    # Convert to indices
                    negative_indexes = list(map(lambda movie_id: user_movie_matrix.columns.get_loc(movie_id),
                                                negative_samples))

                    slate_movies.extend(negative_indexes)

                user_index = user_movie_matrix.index.get_loc(user_id)
                response_vector = response_vector.tolist()

                # Shuffling the negative values
                shuffled = list(zip(slate_movies, response_vector))
                np.random.shuffle(shuffled)
                slate_movies, response_vector = zip(*shuffled)

                sample = [user_index,
                          '|'.join(str(e) for e in all_user_interactions_indexes),
                          '|'.join(str(e) for e in slate_movies),
                          '|'.join(str(e) for e in response_vector)]

                all_samples.append(sample)

            pbar.update(1)

    df = pd.DataFrame(all_samples, columns=['User Index', 'User Interactions', 'Slate Movies', 'Response Vector'])
    df.to_csv(save_location, index=False)

    print("Time taken in seconds: ", time.process_time() - start)

    return df
