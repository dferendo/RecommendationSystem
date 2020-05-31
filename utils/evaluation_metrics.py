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

    pass