import pandas as pd
import surprise as sp
import warnings; warnings.simplefilter('ignore')


ratings = pd.read_csv('ratings.csv')
ratings.head(10)
reader = sp.Reader()
data = sp.Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
movies_df = pd.read_csv('movies.csv')
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())
movies_df['genres'] = movies_df.genres.str.split('|')
print(len(movies_df['movieId'].unique().tolist()))
movies_df.head()
movies_df.isna().sum()
movies_df['year'].fillna(0, inplace=True)
movies_with_genres = movies_df.copy(deep=True)
x = []
for index, row in movies_df.iterrows():
    x.append(index)
    for genre in row['genres']:
        movies_with_genres.at[index, genre] = 1
movies_with_genres.fillna(0.0, inplace=True)
movies_with_genres.drop('genres', axis=1, inplace=True)
movies_with_genres.head()


def get_user_movie_ratings(uid):
    user_ratings = ratings[ratings['userId'] == uid]
    user_movie_ratings = pd.merge(movies_df, user_ratings, on='movieId')[['movieId', 'title', 'rating']]
    return user_movie_ratings
def get_user_genres(uid):
    user_movie_ratings = get_user_movie_ratings(uid)
    user_genres = movies_with_genres[movies_with_genres['movieId'].isin(user_movie_ratings['movieId'])]
    user_genres.reset_index(drop=True, inplace=True)
    user_genres.drop(['movieId', 'title', 'year'], axis=1, inplace=True)
    return user_genres
def content_based_recommender(uid):
    user_movie_ratings = get_user_movie_ratings(uid)
    user_genres_df = get_user_genres(uid)
    user_profile = user_genres_df.T.dot(user_movie_ratings['rating'])
    genres_df = movies_with_genres.copy(deep=True).set_index(movies_with_genres['movieId']).drop(['movieId', 'title', 'year'], axis=1)
    recommendation_df = (genres_df.dot(user_profile)) / user_profile.sum()
    return recommendation_df


ab = ratings['userId'].unique()
final_dict={}
for i in ab:
    rr = content_based_recommender(i)
    final_dict[i] = rr
df = pd.DataFrame.from_dict(final_dict)
df_min_max_scaled = df.copy()
for column in df_min_max_scaled.columns:
    df_min_max_scaled[column] = ((df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (df_min_max_scaled[column].max() - df_min_max_scaled[column].min())*4)+1
print(df_min_max_scaled)
df = df_min_max_scaled.transpose()
print(df)
df.to_csv("content_pivot.csv")
