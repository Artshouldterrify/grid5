import pandas as pd
import calendar
import time


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
    print(user_df)
    for item in user_df.keys():
        if int(item) not in to_ignore:
            rank_dict[item] = user_df[item]
    rank_dict = dict(sorted(rank_dict.items(), key=lambda x: x[1], reverse=True))
    i = 0
    recommended = list()
    for key in rank_dict:
        if i >= 10:
            break
        recommended.append(int(key))
        i += 1
    print(recommended)
    recommended = set(recommended)
    return recommended


# main function
pivot_df = pd.read_csv("pivot.csv", index_col=0)
print(pivot_df)
movie_details_df = pd.read_csv("movies.csv")
ratings_df = pd.read_csv("ratings.csv")
user_list = set(ratings_df['userId'].unique().tolist())
movie_list = set(ratings_df['movieId'].unique().tolist())
while True:
    user = int(input("Enter User ID: "))
    if user in user_list:
        print("Successful login as user", user)
        while True:
            print("Options:")
            print("1. Add Review    2. Get recommendations   3. Print previously rated movies    4. Exit")
            choice = int(input("Enter your choice: "))
            if choice in [1, 2, 3]:
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
ratings_df.to_csv("new_ratings.csv", index=False)
