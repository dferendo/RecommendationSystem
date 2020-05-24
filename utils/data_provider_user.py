# def split_by_user():
#     """
#     # TODO: Should we split by user
#     :return:
#     """
#     df_interactions_sorted = df_sorted_by_timestamp.groupby(['userId'])['movieId'].apply(list)
#
#     user_ids_indexes = []
#     test_interactions = []
#     train_interactions = []
#
#     # TODO: More efficiently?
#     # Splitting the dataset by time-step instead of random
#     for user_id, user_interactions in df_interactions_sorted.items():
#         user_interactions_train = user_interactions[:-SLATE_SIZE]
#         user_interactions_test = user_interactions[-SLATE_SIZE:]
#
#         user_ids_indexes.append(user_id)
#         train_interactions.append(user_interactions_train)
#         test_interactions.append(user_interactions_test)
#
#     ser_train = pd.Series(train_interactions, index=user_ids_indexes)
#     ser_test = pd.Series(test_interactions, index=user_ids_indexes)
#
#     print(ser_train)
#     print(ser_test)


# movies_interactions_counts = df_all.groupby(['movieId']).count()
# # For each interaction, check whether the movieId occurred more than MINIMUM_MOVIE_INTERACTION times
# df_all = df_all.loc[df_all['movieId'].isin(movies_interactions_counts[movies_interactions_counts['timestamp'] >=
#                                                                       MINIMUM_MOVIE_INTERACTION].index)]