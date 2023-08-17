import surprise as srp
import pandas as pd

class SimiliarityRecommender:
    def __init__(self, data_df, user_list, item_list):
        self.data_df = data_df
        self.user_list = user_list
        self.item_list = item_list

    def make_prediction_pivot(self):
        reader = srp.Reader(rating_scale=(1, 5))
        data = srp.Dataset.load_from_df(self.data_df[['userId', 'movieId', 'rating']], reader)
        data_train = data.build_full_trainset()
        sim_options = {
            "name": "cosine",
            "user-based": True
        }
        algo_KNN = srp.KNNBasic(sim_options=sim_options)
        algo_KNN.fit(data_train)
        matrix = list()
        for user in self.user_list:
            temp = list()
            for movie in self.item_list:
                r = algo_KNN.predict(user, movie).est
                temp.append(r)
            matrix.append(temp)
        pivot_df_KNN = pd.DataFrame(matrix, index=self.user_list, columns=self.item_list)
        return pivot_df_KNN
