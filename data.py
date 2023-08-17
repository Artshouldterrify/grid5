import pandas as pd
import numpy as np

def sample(chunk, rate=0.2):
    n = max(int(len(chunk)*rate), 1)
    return chunk.sample(n=n, replace=True, random_state=1)


df = pd.read_csv("ratings.csv")
item_list = df['movieId'].unique().tolist()
user_list = df['userId'].unique().tolist()
df_sampled = df.groupby(['userId'], group_keys=False).apply(sample)
item_list_s = df_sampled['movieId'].unique().tolist()
user_list_s = df_sampled['userId'].unique().tolist()
print(len(df), len(df_sampled))
print(len(item_list), len(item_list_s))
print(len(user_list), len(user_list_s))


