import numpy as np


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


def category_coverage(predicted_slates, movies_categories):
    """
    This ASSUMES that the predicted slates and in index form (ie index of the movie).
    :param predicted_slates:
    :param movies_categories:
    :return:
    """
    cat_coverage = 0.0

    for predicted_slate in list(predicted_slates):
        slate_genres = []

        for movie_index in predicted_slate:
            # Since movie_index is a tensor
            movie_index = movie_index.item()

            movie_genres = movies_categories.iloc[movie_index]

            slate_genres.extend(movie_genres[movie_genres == 1].index.tolist())

        unique_genres = set(slate_genres)
        cat_coverage += (len(unique_genres) / (1.0 * len(movies_categories.columns)))

    return cat_coverage / (1.0 * predicted_slates.size(0))
