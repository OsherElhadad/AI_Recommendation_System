# Osher Elhadad 318969748
import time
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


class Recommender:
    def __init__(self, strategy='user'):
        self.strategy = strategy
        self.similarity = np.NaN
        self.user_item_matrix = None
        self.user_ratings_mean = None
        self.pred = None

    def fit(self, matrix):
        self.user_item_matrix = matrix

        if self.strategy == 'user':

            # User - User based collaborative filtering
            start_time = time.time()

            # Calculate the mean of ratings by users
            self.user_ratings_mean = self.user_item_matrix.mean(axis=1).to_numpy().reshape(-1, 1)

            # Normalize ratings by subtracting the mean of ratings by user and adding epsilon
            normalized_ratings = (self.user_item_matrix.to_numpy() - self.user_ratings_mean + 0.001)

            # Put 0 in every NaN value
            normalized_ratings[np.isnan(normalized_ratings)] = 0

            # Compute the similarity matrix of our model
            self.similarity = 1 - pairwise_distances(normalized_ratings, metric='cosine')

            # Compute the predictions matrix of our model
            pred_matrix = (self.user_ratings_mean + (np.matmul(self.similarity, normalized_ratings) /
                                                     np.array([np.abs(self.similarity).sum(axis=1)]).T)).round(2)
            self.pred = pd.DataFrame(pred_matrix, columns=self.user_item_matrix.columns,
                                     index=self.user_item_matrix.index)

            time_taken = time.time() - start_time
            print('User Model in {} seconds'.format(time_taken))

            return self

        elif self.strategy == 'item':

            # Item - Item based collaborative filtering
            start_time = time.time()

            # Calculate the mean of ratings by users
            self.user_ratings_mean = self.user_item_matrix.mean(axis=1).to_numpy().reshape(-1, 1)

            # Normalize ratings by subtracting the mean of ratings by user and adding epsilon
            normalized_ratings = (self.user_item_matrix.to_numpy() - self.user_ratings_mean + 0.001)

            # Put 0 in every NaN value
            normalized_ratings[np.isnan(normalized_ratings)] = 0

            # Compute the similarity matrix of our model
            self.similarity = 1 - pairwise_distances(normalized_ratings.T, metric='cosine')

            # Compute the predictions matrix of our model
            pred_matrix = (self.user_ratings_mean + (np.matmul(normalized_ratings, self.similarity) /
                                                     np.array([np.abs(self.similarity).sum(axis=1)]))).round(2)
            self.pred = pd.DataFrame(pred_matrix, columns=self.user_item_matrix.columns,
                                     index=self.user_item_matrix.index)

            time_taken = time.time() - start_time
            print('Item Model in {} seconds'.format(time_taken))

            return self

    def recommend_items(self, user_id, k=5):

        # Get a numpy array of all predicted user ratings
        predicted_ratings_of_user = self.pred.T[user_id].to_numpy()

        # Get a numpy array of all user ratings
        ratings_of_user = self.user_item_matrix.T[user_id].to_numpy()

        # Put 0 in every already rated products, for taking them as a last option after sorting
        predicted_ratings_of_user[~np.isnan(ratings_of_user)] = 0

        # Return the top k recommended products
        top_k_indexes_of_ratings = np.argsort(-predicted_ratings_of_user, kind='mergesort')[0:k]
        return self.pred.T[user_id].iloc[top_k_indexes_of_ratings].index.to_list()
