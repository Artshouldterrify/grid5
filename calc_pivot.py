import pandas as pd
from sklearn.model_selection import train_test_split
from similarityRecommmender import SimiliarityRecommender
from popularityRecommender import PopularRecommender
from collabRecommender import CollabRecommender
from evaluation import Evaluator

def combine_pivots(pivot_list, weight_list):
    pivot_list[0] = (pivot_list[0]*weight_list[0] + pivot_list[1]*weight_list[1] + pivot_list[2]*weight_list[2] + pivot_list[3]*weight_list[3])/sum(weight_list)
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


df = pd.read_csv("data/ratings.csv")
item_list = df['movieId'].unique().tolist()
item_list = sorted(item_list)
user_list = df['userId'].unique().tolist()
df_train, df_test = train_test_split(df, train_size=0.8, random_state=42)
train_item_list = df_train['movieId'].unique().tolist()
train_user_list = df_train['userId'].unique().tolist()
c = CollabRecommender(df_train, train_user_list, train_item_list)
c_pivot = c.make_prediction_pivot(18)
c_pivot = augument_pivot(c_pivot, item_list, user_list)
e = Evaluator(test_df=df_test, pivot=c_pivot)
p = PopularRecommender(df_train, train_user_list, train_item_list)
p_pivot = p.make_prediction_pivot()
p_pivot = augument_pivot(p_pivot, item_list, user_list)
e1 = Evaluator(test_df=df_test, pivot=p_pivot)
s = SimiliarityRecommender(df_train, train_user_list, train_item_list)
s_pivot = s.make_prediction_pivot()
s_pivot = augument_pivot(s_pivot, item_list, user_list)
e2 = Evaluator(test_df=df_test, pivot=s_pivot)
con_pivot = pd.read_csv("data/content_pivot.csv", index_col=0)
con_pivot.columns = con_pivot.columns.astype(int)
con_pivot = con_pivot[item_list]
e3 = Evaluator(test_df=df_test, pivot=con_pivot)
print(e.precision_at_k(10), e1.precision_at_k(10), e2.precision_at_k(10), e3.precision_at_k(10))
print(e.RMSE(), e1.RMSE(), e2.RMSE(), e3.RMSE())
best_val = 0.0
best_vector = list()
weight_list = [100, 10, 10, 10]
pivot_list = [c_pivot, s_pivot, p_pivot, con_pivot]
pivot = combine_pivots(pivot_list, weight_list)
pivot.to_csv("data/pivot.csv")
