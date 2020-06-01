from utils.arg_parser import extract_args_from_json
from utils.data_provider import split_dataset, load_movie_categories
from utils.reset_seed import set_seeds
import numpy as np
import os
import time


def generate_slate_formation(row_interactions, user_movie_matrix, slate_size, negative_sampling_for_slates):
    all_movies_that_can_be_sampled = np.array(user_movie_matrix.columns)

    grouped_users = row_interactions.groupby(['userId'])['movieId'].apply(list)

    all_samples = []

    for user_id, user_interactions in grouped_users.items():
        # Get the possible index of movieIds that we can sample for this user
        movies_with_no_interactions_with_user = np.setxor1d(all_movies_that_can_be_sampled, user_interactions)

        for negative_samples_amount in negative_sampling_for_slates:
            assert negative_samples_amount <= slate_size

            slate_movies = []
            response_vector = np.zeros(slate_size, dtype=np.int32)
            all_user_interactions = user_movie_matrix.loc[user_id].to_list()

            positive_samples_amount = slate_size - negative_samples_amount

            if positive_samples_amount != 0:
                positive_samples = user_interactions[-positive_samples_amount:]

                response_vector[:positive_samples_amount] = 1

                # Convert to indices
                positive_indexes = list(map(lambda movie_id: user_movie_matrix.columns.get_loc(movie_id),
                                            positive_samples))

                slate_movies.extend(positive_indexes)

                # Remove positive samples that are in the slate from user interactions
                for positive_index in positive_indexes:
                    all_user_interactions[positive_index] = 0

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

            sample = [user_index]

            sample.extend(all_user_interactions)
            sample.extend(slate_movies)
            sample.extend(response_vector)

            all_samples.append(sample)

    return np.vstack(all_samples)


if __name__ == '__main__':
    print("Generating slate formation.....")
    configs = extract_args_from_json()
    set_seeds(configs['seed'])

    start = time.process_time()

    df_train, df_val, df_test, df_train_matrix, df_val_matrix, df_test_matrix = split_dataset(configs)
    movies_categories = load_movie_categories(configs)

    samples = generate_slate_formation(df_train, df_train_matrix, configs['slate_size'],
                                       configs['negative_sampling_for_slates'])

    save_file = os.path.join(configs['save_location'], 'train_sf_{}.npy'.format(configs['slate_size']))

    print("Time taken in seconds: ", time.process_time() - start)

    np.save(save_file, samples.astype(int))
