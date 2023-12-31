import pandas as pd
import numpy as np
import calendar
import time
from scipy import spatial


def add_review(rate_df, user_id, movies_list):
    movie_id = int(input("Movie for which review is to be added:"))
    if movie_id not in movies_list:
        print("Invalid movie id.")
        return
    rate = int(input("Enter movie rating: "))
    movie = rate_df[(rate_df['userId'] == user_id) & (rate_df['movieId'] == movie_id)]
    if(len(movie)) >= 1:
        rate_df.loc[(rate_df['userId'] == user_id) & (rate_df['movieId'] == movie_id)]['rating'] = rate
        return
    gmt = time.gmtime()
    new_row = pd.DataFrame({'userId': user_id, 'movieId': movie_id, 'rating': rate, 'timestamp': calendar.timegm(gmt)},
                           index=[len(rate_df)])
    rate_df = pd.concat([rate_df, new_row], axis=0)
    print(rate_df)
    return rate_df


def recommend(pivot, user_id, to_ignore):
    user_df = pivot.loc[user_id]
    rank_dict = dict()
    for item in user_df.keys():
        if int(item) not in to_ignore:
            rank_dict[item] = user_df[item]
    rank_dict = dict(sorted(rank_dict.items(), key=lambda x: x[1], reverse=True))
    i = 0
    recommended = list()
    weights = list()
    for key in rank_dict:
        if i >= 50:
            break
        recommended.append(int(key))
        weights.append(rank_dict[key])
        i += 1
    weights /= sum(weights)
    recommended = np.random.choice(recommended, size=10, replace=False, p=weights)
    recommended = set(recommended)
    return recommended


# main function
pivot_df = pd.read_csv("data/pivot.csv", index_col=0)
movie_details_df = pd.read_csv("data/movies.csv")
ratings_df = pd.read_csv("data/ratings.csv")
user_list = set(ratings_df['userId'].unique().tolist())
movie_list = set(ratings_df['movieId'].unique().tolist())
while True:
    user = int(input("Enter User ID: "))
    if user in user_list:
        print("Successful login as user", user)
        while True:
            print("Options:")
            print("1. Add Review    2. Get recommendations   3. Print previously rated movies    4. Browse an item   5. Exit")
            choice = int(input("Enter your choice: "))
            if choice in [1, 2, 3, 4]:
                if choice == 1:
                    ratings_df = add_review(ratings_df, user, movie_list)
                elif choice == 2:
                    movies_rated = set(ratings_df.loc[ratings_df['userId'] == user]['movieId'].unique().tolist())
                    recommendations = recommend(pivot_df, user, movies_rated)
                    print("Your recommendations:")
                    for it in recommendations:
                        print(movie_details_df.loc[movie_details_df['movieId'] == it].to_dict())
                elif choice == 3:
                    movies_rated = set(ratings_df.loc[ratings_df['userId'] == user]['movieId'].unique().tolist())
                    for it in movies_rated:
                        print(movie_details_df.loc[movie_details_df['movieId'] == it].to_dict())
                elif choice == 4:
                    item_code = int(input("Enter item ID: "))
                    print("Viewing movie:")
                    print(movie_details_df.loc[movie_details_df['movieId'] == item_code].to_dict())
                    item_vec = pivot_df[str(item_code)].values.tolist()
                    item_vectors = dict()
                    item_list = pivot_df.columns.values.tolist()
                    for i in range(len(pivot_df.iloc[0])):
                        vec = pivot_df.iloc[:, i].values.tolist()
                        item_vectors[item_list[i]] = 1 - spatial.distance.cosine(vec, item_vec)
                    item_vectors = dict(sorted(item_vectors.items(), key=lambda x: x[1], reverse=True))
                    recommended = list()
                    weights = list()
                    i = 0
                    for key in item_vectors:
                        if i >= 50:
                            break
                        recommended.append(int(key))
                        weights.append(item_vectors[key])
                        i += 1
                    weights /= sum(weights)
                    recommended = np.random.choice(recommended, size=10, replace=False, p=weights)
                    recommended = set(recommended)
                    print("Your recommendations:")
                    for it in recommended:
                        print(movie_details_df.loc[movie_details_df['movieId'] == it].to_dict())
            elif choice == 5:
                break
            else:
                print("Invalid Choice.")
    else:
        print("Invalid User ID")
    ch = input("Do you want to continue? (y/n)  ")
    if ch == "y" or ch == "Y":
        continue
    else:
        break
ratings_df.to_csv("data/ratings.csv", index=False)
