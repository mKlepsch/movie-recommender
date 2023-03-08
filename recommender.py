## metafunctions fpr the recommendations
import pandas as pd
import numpy as np
from utils import NMF,movie_title_2_id,get_cosine_similertiy,get_unseen_movies,get_similar_user,get_recommended_movies,get_ID_to_title,recommend_nmf_new_user,get_top_rated_movies



def recommend_with_cossin(query:dict,original_DF:pd.core.frame.DataFrame,movie_df:pd.core.frame.DataFrame,n:int=3):
    temp = { str(movie_title_2_id(title=title,movie_df=movie_df)):int(rating) for title, rating in query.items()}
    df_with_new_user, cosine_similarity_matrtix = get_cosine_similertiy(new_user_query=temp, original_DF=original_DF)
    unseen_movies = get_unseen_movies(data= df_with_new_user)
    close_users = get_similar_user(similiarity_matrix = cosine_similarity_matrtix, n=10)
    movies_recs = get_recommended_movies(unseen_movies=unseen_movies,closes_users=close_users,data=df_with_new_user,similiarity_matrix=cosine_similarity_matrtix,n_movies=n)
    recs = get_ID_to_title(movie_df=movie_df,movieIds=movies_recs['movie'])
    return recs


### random baseline

def recommender_random(movie_list:list,n:int=3):
    return movie_list.sample(n).to_list()

### recommend NMF

def recommender_with_NMF(query:dict,model:NMF,q_matrix:pd.core.frame.DataFrame,imputed_values:pd.core.series.Series,movie_df:pd.core.frame.DataFrame,n:int=3):
    r_new_user = recommend_nmf_new_user(user_query=query,
                                        q_matrix=q_matrix,
                                        imputed_values=imputed_values,
                                        nmf_model=model)
    top10 = get_top_rated_movies(r_new_user,n_top=n)
    movie_recs = get_ID_to_title(movie_df=movie_df,movieIds=top10.index)
    return movie_recs