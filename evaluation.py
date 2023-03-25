# Osher Elhadad 318969748
import numpy as np
from sklearn.metrics import mean_squared_error


def RMSE(test_set, cf):

    # Create a vector of ratings from our model predictions metrix
    pred_ratings = np.zeros(len(test_set))

    # Create a vector of ratings from our benchmark: user ratings mean matrix
    benchmark_mean_ratings = np.zeros(len(test_set))

    # Create a vector of ratings from the test set
    test_real_ratings = test_set.iloc[:, 2].values

    # Fill pred_ratings from our predictions model, and benchmark_mean_ratings from user_ratings_mean
    for i, row in test_set.iterrows():
        pred_ratings[i] = cf.pred[row.ProductId][row.UserId]
        benchmark_mean_ratings[i] = cf.user_ratings_mean[np.where(cf.user_item_matrix.index.to_numpy() == row.UserId)]

    # Calculate the RMSE between test_real_ratings and pred_ratings
    print(f"RMSE {cf.strategy}-based CF: {round(np.sqrt(mean_squared_error(test_real_ratings, pred_ratings)), 5)}")

    # Calculate the RMSE between test_real_ratings and benchmark_mean_ratings
    print(f"RMSE benchmark: {round(np.sqrt(mean_squared_error(test_real_ratings, benchmark_mean_ratings)), 5)}")


def metric_at_k(test_set, cf, k, metric):
    users_pred_metric_sum = 0
    users_benchmark_metric_sum = 0
    users_metric_count = 0
    if metric == 'Precision':
        divide_by = k

    # We take the top k products that have the highest mean
    top_k_products_mean_indexes = np.argsort(-cf.user_item_matrix.mean(axis=0).to_numpy().reshape(-1), kind='mergesort')[0:k]
    top_k_products_benchmark = cf.user_item_matrix.T.iloc[top_k_products_mean_indexes].index.to_list()

    # Calculate the metric for every user
    for unique_user in test_set['UserId'].unique():

        # Relevant items are items of this user and has rating of 3 and above
        records_of_user = test_set.loc[test_set['UserId'] == unique_user]
        unique_relevant_user_items = records_of_user.loc[records_of_user['Rating'] >= 3, 'ProductId'].unique()
        if len(unique_relevant_user_items) > 0:
            top_k_recommended_items = cf.recommend_items(unique_user, k)
            if metric == 'Recall':
                divide_by = len(unique_relevant_user_items)
            users_pred_metric_sum += len(set(unique_relevant_user_items).
                                         intersection(set(top_k_recommended_items))) / divide_by
            users_benchmark_metric_sum += len(set(unique_relevant_user_items).
                                              intersection(set(top_k_products_benchmark))) / divide_by
            users_metric_count += 1

    print(f"{metric} for top {k} {cf.strategy}-based CF: {round((users_pred_metric_sum / users_metric_count), 5)}")
    print(f"{metric} for top {k} benchmark: {round((users_benchmark_metric_sum / users_metric_count), 5)}")


def precision_at_k(test_set, cf, k):
    metric_at_k(test_set, cf, k, 'Precision')


def recall_at_k(test_set, cf, k):
    metric_at_k(test_set, cf, k, 'Recall')
