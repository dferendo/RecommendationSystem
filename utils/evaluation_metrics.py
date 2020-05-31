import numpy as np
"""
TODO: Vectorize the following functions
"""


def hit_ratio(predicted_slates, ground_truths):
    recall_score = 0.0

    for predicted_slate, ground_truths_slate in list(zip(predicted_slates, ground_truths)):
        intersection = np.intersect1d(predicted_slate, ground_truths_slate)

        recall_score += (len(intersection) / (1.0 * len(ground_truths_slate)))

    return recall_score / (1.0 * predicted_slates.size(0))


def precision(predicted_slates, ground_truths):
    precision_score = 0.0

    for predicted_slate, ground_truths_slate in list(zip(predicted_slates, ground_truths)):
        intersection = np.intersect1d(predicted_slate, ground_truths_slate)

        precision_score += (len(intersection) / (1.0 * len(predicted_slate)))

    return precision_score / (1.0 * predicted_slates.size(0))


def category_coverage(predicted_slates, train_loader):
    """

    :param predicted_slates:
    :param train_loader: The is needed so that we can convert movie indexes to movie ids to extract categories
    :return:
    """
    category_coverage = 0.0

    for predicted_slate in list(predicted_slates):
        movie_ids = np.array(list(map(lambda movie_index: train_loader.dataset.train_matrix.columns[movie_index],
                                      predicted_slate)))

        movie_category = train_loader.dataset.movies_categories

        slate_genres = []

        for movie_id in movie_ids:
            movie_genres = movie_category.loc[movie_id]

            slate_genres.extend(movie_genres[movie_genres == 1].index.tolist())

        unique_genres = set(slate_genres)

        category_coverage += (len(unique_genres) / (1.0 * len(movie_category.columns)))

    return category_coverage / (1.0 * predicted_slates.size(0))
