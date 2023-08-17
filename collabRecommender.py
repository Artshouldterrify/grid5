import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import surprise as srp

class CollabRecommender:
    def __init__(self, data_df, user_list, item_list):
        self.data_df = data_df
        self.user_list = user_list
        self.item_list = item_list

    def make_prediction_pivot(self):
        pivot_df = self.data_df.pivot(index='userId', columns='movieId', values='rating').fillna(0.0)
        pivot_matrix = pivot_df.values
        sparse_pivot = csr_matrix(pivot_matrix)
        ids = list(pivot_df.index)
        A, B, C = svds(sparse_pivot, 20)
        B = np.diag(B)
        prediction_matrix = np.dot(np.dot(A, B), C)
        prediction_matrix = ((prediction_matrix - prediction_matrix.min()) / (
                    prediction_matrix.max() - prediction_matrix.min())) * 4 + 1
        prediction_df = pd.DataFrame(prediction_matrix, columns=pivot_df.columns, index=ids)
        return prediction_df

    def make_prediction_pivot_using_surprise(self):
        reader = srp.Reader(rating_scale=(1, 5))
        data = srp.Dataset.load_from_df(self.data_df[['userId', 'movieId', 'rating']], reader)
        data_train = data.build_full_trainset()
        algo = srp.SVD(n_factors=20)
        algo.fit(data_train)
        matrix = list()
        for user in self.user_list:
            temp = list()
            for movie in self.item_list:
                r = algo.predict(user, movie).est
                temp.append(r)
            matrix.append(temp)
        pivot_df = pd.DataFrame(matrix, index=self.user_list, columns=self.item_list)
        return pivot_df

