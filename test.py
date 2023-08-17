import pandas as pd
import random
from sklearn.model_selection import train_test_split
from similarityRecommmender import SimiliarityRecommender
from popularityRecommender import PopularRecommender
from collabRecommender import CollabRecommender
from evaluation import Evaluator

def combine_pivots(pivot_list, weight_list):
    pivot_list[0] = (pivot_list[0]*weight_list[0] + pivot_list[1]*weight_list[1] + pivot_list[2]*weight_list[2])/sum(weight_list)
    return pivot_list[0]


def augument_pivot(pivot, items_list, users_list):
    items_in_pivot = set(pivot.columns.values.tolist())
    items_not_in_pivot = set(items_list) - items_in_pivot
    users_in_pivot = set(pivot.index.values.tolist())
    users_not_in_pivot = set(users_list) - users_in_pivot
    mean_item_rating, mean_user_rating = 0.0, 0.0
    for i in range(len(pivot)):
        mean_user_rating += pivot.iloc[i, :].mean()
    mean_user_rating /= len(pivot)
    for j in range(len(pivot.columns)):
        mean_item_rating += pivot.iloc[:, j].mean()
    mean_item_rating /= len(pivot.columns)
    new_df = pd.DataFrame(index=pivot.index, columns=list(items_not_in_pivot)).fillna(mean_item_rating)
    pivot = pd.concat([new_df, pivot], axis=1)
    for user in users_not_in_pivot:
        pivot.loc[user] = [mean_user_rating for x in range(len(pivot.columns))]
    pivot = pivot.reindex(sorted(pivot.index), axis=0)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    return pivot


df = pd.read_csv("ratings.csv")
print(df)
item_list = df['movieId'].unique().tolist()
user_list = df['userId'].unique().tolist()
df_train, df_test = train_test_split(df, train_size=0.8)
train_item_list = df_train['movieId'].unique().tolist()
train_user_list = df_train['userId'].unique().tolist()
c = CollabRecommender(df_train, train_user_list, train_item_list)
c_pivot = c.make_prediction_pivot()
c_pivot = augument_pivot(c_pivot, item_list, user_list)
print(c_pivot)
# p = PopularRecommender(df_train, train_user_list, train_item_list)
# p_pivot = p.make_prediction_pivot()
# p_pivot = augument_pivot(p_pivot, item_list, user_list)
# print(p_pivot)
# s = SimiliarityRecommender(df_train, train_user_list, train_item_list)
# s_pivot = s.make_prediction_pivot()
# s_pivot = augument_pivot(s_pivot, item_list, user_list)
# print(s_pivot)
e = Evaluator(test_df=df_test, pivot=c_pivot)
print(e.recall_at_k(10))
# e2 = Evaluator(test_df=df_test, pivot=p_pivot)
# e3 = Evaluator(test_df=df_test, pivot=s_pivot)
# print("PAK:", e.precision_at_k(10), "RMSE:", e.RMSE())
# print("PAK pop:", e2.precision_at_k(10), "RMSE:", e2.RMSE())
# print("PAK similar:", e3.precision_at_k(10), "RMSE:", e3.RMSE())
# combined_pivot = combine_pivots([c_pivot, p_pivot, s_pivot], [100, 5, 1])
# e4 = Evaluator(test_df=df_test, pivot=combined_pivot)
# print("PAK combined:", e4.precision_at_k(10), "RMSE:", e4.RMSE())
# combined_pivot.to_csv('pivot.csv')
