import random

import pandas as pd
import numpy as np


class Evaluator:
    def __init__(self, test_df, pivot):
        self.pivot = pivot
        self.test_df = test_df

    def RMSE(self):
        MSE = 0.0
        for i in range(len(self.test_df)):
            actual_rating = self.test_df.iloc[i]['rating']
            user, item = self.test_df.iloc[i]['userId'], self.test_df.iloc[i]['movieId']
            predicted_rating = self.pivot.loc[user, item]
            MSE += (actual_rating - predicted_rating) ** 2
        MSE /= len(self.test_df)
        return np.sqrt(MSE)

    def mean_precision_at_k(self, k):
        result = 0.0
        test_user_list = self.test_df['userId'].unique().tolist()
        for user in test_user_list:
            user_df = self.pivot.loc[user]
            rank_dict = dict()
            for item in user_df.keys():
                rank_dict[item] = user_df[item]
            interacted_items = self.test_df[self.test_df['userId'] == user]['movieId'].values.tolist()
            d = min(k, len(interacted_items))
            rank_dict = dict(sorted(rank_dict.items(), key=lambda x: x[1], reverse=True))
            i = 0
            recommended = list()
            for key in rank_dict:
                if i >= d:
                    break
                recommended.append(key)
                i += 1
            recommended = set(recommended)
            relevant = 0
            for item in interacted_items:
                if item in recommended:
                    relevant += 1
            precision = relevant / len(interacted_items)
            result += precision
        return result / len(test_user_list)

    def precision_at_k(self, k):
        relevant = 0
        test_user_list = self.test_df['userId'].unique().tolist()
        recommendation_dict = dict()
        for user in test_user_list:
            user_df = self.pivot.loc[user]
            rank_dict = dict()
            for item in user_df.keys():
                rank_dict[item] = user_df[item]
            rank_dict = dict(sorted(rank_dict.items(), key=lambda x: x[1], reverse=True))
            i = 0
            recommended = list()
            for key in rank_dict:
                if i >= k:
                    break
                recommended.append(int(key))
                i += 1
            recommended = set(recommended)
            recommendation_dict[user] = recommended
        for i in range(len(self.test_df)):
            user = self.test_df.iloc[i]['userId']
            recommended = recommendation_dict[user]
            if self.test_df.iloc[i]['movieId'] in recommended:
                relevant += 1
        return relevant/len(self.test_df)

    def recall_at_k(self, k):
        test_user_list = self.test_df['userId'].unique().tolist()
        result = 0.0
        test_movie_list = self.test_df['movieId'].unique().tolist()
        rec_dict = dict()
        for user in test_user_list:
            user_df = self.pivot.loc[user]
            rank_dict = dict()
            for item_key in user_df.keys():
                rank_dict[item_key] = user_df[item_key]
            rank_list = list(sorted(rank_dict.items(), key=lambda x: x[1], reverse=True))
            rec_dict[user] = rank_list
        for user in test_user_list:
            interacted_items = self.test_df[self.test_df['userId'] == user]['movieId'].values.tolist()
            interacted_items = set(interacted_items)
            not_interacted_items = set(test_movie_list) - interacted_items
            relevant = 0
            for item in interacted_items:
                random_items = set(random.sample(list(not_interacted_items), k=100))
                random_items = random_items.union({item})
                rank_list = rec_dict[user]
                rank_this = list()
                for it in rank_list:
                    if it[0] in random_items:
                        rank_this.append(it[0])
                rank_this = rank_this[:k]
                if item in rank_this:
                    relevant += 1
            relevant /= len(interacted_items)
            result += relevant
        result /= len(test_user_list)
        return result



