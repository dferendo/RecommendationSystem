import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from pandas.api.types import CategoricalDtype
import os
import re


def split_it(year):
    return re.findall('(\d+)', year)[-1]


def get_sparse_df(df, all_movies_in_train):
    """
    Note: This process is done to conserve memory. If memory is not an issue, simply use pivot
    (Ie df.pivot(index='userId', columns='movieId', values='rating').fillna(0))
    :param all_movies_in_train: This is needed so that the train/test will have the same amount of columns
    :param df:
    :return:
    """
    users_category = CategoricalDtype(sorted(df['userId'].unique()), ordered=True)
    movies_category = CategoricalDtype(sorted(all_movies_in_train), ordered=True)

    row = df['userId'].astype(users_category).cat.codes
    col = df['movieId'].astype(movies_category).cat.codes

    sparse_matrix = csr_matrix((df["rating"], (row, col)),
                               shape=(users_category.categories.size, movies_category.categories.size))

    sparse_df = pd.DataFrame.sparse.from_spmatrix(sparse_matrix, index=users_category.categories,
                                                  columns=movies_category.categories)

    return sparse_df


def load_movie_categories(configs, all_movies_in_train):
    movies_categories_location = os.path.join(configs['data_location'], 'movies.csv')

    # Loading
    df_all = pd.read_csv(movies_categories_location, dtype={'movieId': np.int32, 'title': np.str, 'genres': np.str})
    df_titles = pd.read_csv(movies_categories_location, dtype={'movieId': np.int32, 'title': np.str})

    # A movie can have multiple genres and each genre is seperate by a '|'
    df_all['genres'] = df_all['genres'].str.split('|')
    df_titles['title'] = df_titles['title'].apply(split_it)

    # Explode transforms a list to a row, thus for each movie that have multiple genres, create a new row
    df_all = df_all.explode('genres')
    df_all['hit'] = 1

    genres_unique = df_all['genres'].unique()

    # Keep only the movies that are found in the training examples so that the indexes match
    df_all = df_all[df_all['movieId'].isin(all_movies_in_train)]
    df_titles = df_titles[df_titles['movieId'].isin(all_movies_in_train)]

    # Convert to a 2-d matrix
    movies_category = CategoricalDtype(sorted(all_movies_in_train), ordered=True)
    genres_category = CategoricalDtype(sorted(genres_unique), ordered=True)

    row = df_all['movieId'].astype(movies_category).cat.codes
    col = df_all['genres'].astype(genres_category).cat.codes

    sparse_matrix = csr_matrix((df_all['hit'], (row, col)),
                               shape=(movies_category.categories.size, genres_category.categories.size))

    sparse_df = pd.DataFrame.sparse.from_spmatrix(sparse_matrix, index=movies_category.categories,
                                                  columns=genres_category.categories)

    # Drop this column
    if '(no genres listed)' in sparse_df:
        sparse_df = sparse_df.drop('(no genres listed)', 1)

    return sparse_df.values, df_titles['title'].values


def split_dataset(configs):
    ratings_location = os.path.join(configs['data_location'], 'ratings.csv')

    # Loading
    df_all = pd.read_csv(ratings_location, dtype={'userId': np.int32, 'movieId': np.int32})

    # Make dataset implicit (ie User had interaction/User did not have interaction)
    df_all = df_all[df_all['rating'] >= configs['implicit_rating']]
    df_all.loc[df_all['rating'] >= configs['implicit_rating'], 'rating'] = 1

    if configs['minimum_movie_interaction'] != -1:
        movies_interactions_counts = df_all.groupby(['movieId']).count()

        # For each interaction, check whether the userId occurred more than MINIMUM_MOVIE_INTERACTION times
        df_all = df_all.loc[df_all['movieId'].isin(movies_interactions_counts[movies_interactions_counts['timestamp'] >=
                                                                              configs['minimum_movie_interaction']].index)]

    # Sorting is done so that the model does not have access to future interactions
    df_sorted_by_timestamp = df_all.sort_values(by=['timestamp'])

    test_size = int(len(df_sorted_by_timestamp) * configs['test_set_size'])
    test_indexes_start = len(df_sorted_by_timestamp) - test_size

    df_train, df_test = np.split(df_sorted_by_timestamp, [test_indexes_start])

    # If we are training, we only need validation set
    if configs['is_training']:
        print("Getting dataset for training.....")
        val_size = int(len(df_train) * configs['validation_set_size'])
        val_indexes_start = len(df_train) - val_size

        df_train, df_test = np.split(df_train, [val_indexes_start])
    else:
        print("Getting dataset for testing.....")

    if configs['minimum_user_interaction'] != -1:
        users_interactions_counts = df_train.groupby(['userId']).count()
        # For each interaction, check whether the userId occurred more than MINIMUM_USER_INTERACTION times
        df_train = df_train.loc[df_train['userId'].isin(users_interactions_counts[users_interactions_counts['timestamp'] >=
                                                                                  configs['minimum_user_interaction']].index)]

    # Remove any users that do not appear in the training set
    df_test = df_test.loc[df_test['userId'].isin(df_train['userId'].unique())]

    # Remove any movies that do not appear in the training set from the test set
    df_test = df_test.loc[df_test['movieId'].isin(df_train['movieId'].unique())]

    # This is needed so that the train/test will have the same amount of columns
    all_movies_in_train = df_train['movieId'].unique()

    movies_categories, titles = load_movie_categories(configs, all_movies_in_train)

    return df_train, df_test, get_sparse_df(df_train, all_movies_in_train), \
           get_sparse_df(df_test, all_movies_in_train), movies_categories, titles.astype(np.int32)
