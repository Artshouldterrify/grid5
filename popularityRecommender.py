import pandas as pd

class PopularRecommender:
    def __init__(self, data_df, user_list, item_list):
        self.data_df = data_df
        self.user_list = user_list
        self.item_list = item_list

    def make_prediction_pivot(self):
        popularity_df = self.data_df.groupby('movieId')['rating'].agg(['count', 'mean'])
        popularity_df = popularity_df[popularity_df['count'] >= 100]
        popularity_df['mean'] = (popularity_df['mean'] * (popularity_df['count'] / (popularity_df['count'] + 100))) + \
                                (popularity_df['mean'].mean() * (100 / (100 + popularity_df['count'])))
        temp_list = list()
        for item in self.item_list:
            if item not in popularity_df.index:
                temp_list.append(0.0)
            else:
                temp_list.append(popularity_df.loc[item]['mean'])
        temp_matrix = list()
        for user in self.user_list:
            temp_matrix.append(temp_list)
        popularity_predict_df = pd.DataFrame(temp_matrix, index=self.user_list, columns=self.item_list)
        return popularity_predict_df

