import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
import pickle



def load_movies(movies_path:str):
    '''
    loads the movie file with following Columns movieId,title,genres
    '''
    movies = pd.read_csv(movies_path)
    movies['movieId']=movies['movieId'].astype(str)
    return movies 

def load_data(ratings_path:str, movies_path:str):
    """
    load both csv files with with the user ratings and the movies
    both files need a column with the moveId
    """
    movies = load_movies(movies_path)
    ratings = pd.read_csv(ratings_path)
    ratings['movieId']=ratings['movieId'].astype(str)
    combined_df = ratings.join(other=movies.set_index('movieId'), on='movieId',how='left')
    return combined_df

def get_ID_to_title(movie_df:pd.core.frame.DataFrame, movieIds:list):
    '''
    returns a the title from from IDs if they exist
    needs a DF with the a ID and a title
    '''
    titles = [movie_df[movie_df['movieId']==id]['title'].values[0] for id in movieIds]
    return titles


def movie_title_2_id(title:str,movie_df):
    movid_Title_2_id=dict(zip(movie_df['title'],movie_df['movieId']))
    return int(movid_Title_2_id[title])

#### NMF methods
def get_initial_rating_df(combined_df:pd.core.frame.DataFrame):
        '''
        returns a Dataframe with userId vs MoviesId Table with the ratings in the cells
        '''
        return combined_df.pivot_table(index='userId',columns='movieId',values='rating')

def calcuate_r_matrix(combined_df:pd.core.frame.DataFrame):
    '''
    takes the combined dataframe from ratings and movies
    returns a pivot table: UserID vs movidID with the ratings in the matix
    and returns the mean ratings 4 later user
    fills the missing values with the average value
    '''
    means = combined_df.mean()
    r_matrix = combined_df.fillna(means)

    return r_matrix, means

def create_nmf_model(n_components:int,r_matrix:pd.core.frame.DataFrame):
        '''
        creates a NMF model object based on the number of components and r dataframe
        '''
        nmf_model = NMF(n_components=n_components,max_iter=1000)
        nmf_model.fit(r_matrix)
        return nmf_model

def get_Q_and_P_matrix (nmf_model:NMF,r_matrix:pd.core.frame.DataFrame):
        '''
        returns the Q and the P matrix based on a NMF model and r dataframe
        '''
        Q_matrix = nmf_model.components_
        Q_matrix = pd.DataFrame(data=Q_matrix,
                columns=nmf_model.feature_names_in_,
                index= nmf_model.get_feature_names_out())
        P_matrix = nmf_model.transform(r_matrix)
        P_matrix = pd.DataFrame(data=P_matrix,
                columns=nmf_model.get_feature_names_out(),
                index = r_matrix.index)
        return Q_matrix,P_matrix
        
def get_best_components(r_matrix:pd.core.frame.DataFrame,max_components:int):
        '''
        findes the number of componets with the smalles reconstruction error
        need a r Dataframe and the the maximum number of components you want to search for
        returns a integer for the componets with the smallest error
        '''
        components = pd.DataFrame()
        for i in np.linspace(1, max_components, num=max_components):
                model = create_nmf_model(n_components=int(i),r_matrix=r_matrix)
                components = pd.concat([components,
                pd.DataFrame({'components' : [int(i)],'error':[model.reconstruction_err_]})])
        components.set_index('components').plot()
        #display(components[components['error']==components['error'].min()]) # uncomment if running in juypter notebooker/lab
        return components[components['error']==components['error'].min()]['components'].values[0]
        
def get_r_predtion(P_matrix:pd.core.frame.DataFrame,Q_matrix:pd.core.frame.DataFrame):
        '''
        returns the reconsturced matrix from the P and Q matrix
        '''
        r_hat_matrix = np.dot(P_matrix,Q_matrix)
        return  pd.DataFrame(data=r_hat_matrix,
                columns=Q_matrix.columns,
                index = P_matrix.index)


def save_NMF_and_imputed(nmf_model:NMF, imputed_values:pd.core.series.Series, q_matrix:pd.core.frame.DataFrame,p_matrix:pd.core.frame.DataFrame, path_no_ending:str):
        '''
        saves the model,imputed values,P and Q matrix for later use
        the file path should be without file ending - just the base file
        {pathfile}_imputed.pkl {pathfile}_model.pkl {pathfile}_Q_matrix.pkl {pathfile}_P_matrix.pkl
        '''
        with open(f'{path_no_ending}_q_matrix.pkl',mode='wb') as file:
                pickle.dump(q_matrix,file)
        with open(f'{path_no_ending}_p_matrix.pkl',mode='wb') as file:
                pickle.dump(p_matrix,file)
        with open(f'{path_no_ending}_imputed.pkl',mode='wb') as file:
                pickle.dump(imputed_values,file)
        with open(f'{path_no_ending}_model.pkl',mode='wb') as file:
                pickle.dump(nmf_model,file)

def load_NMF_and_imputed(pathfile):
        '''
        loads a NMF model, the imputed values and the Q and P matrix for it
        give the path without the file ending
        {pathfile}_imputed.pkl {pathfile}_model.pkl {pathfile}_Q_matrix.pkl {pathfile}_P_matrix.pkl
        '''
        with open(f'{pathfile}_q_matrix.pkl','rb') as file:
                q_matrix = pickle.load(file)
        with open(f'{pathfile}_p_matrix.pkl','rb') as file:
                p_matrix = pickle.load(file)
        with open(f'{pathfile}_model.pkl','rb') as file:
                model = pickle.load(file)
        with open(f'{pathfile}_imputed.pkl','rb') as file:
                imputed = pickle.load(file)
        return model, imputed, q_matrix, p_matrix

def recommend_nmf_new_user(user_query:dict,q_matrix:pd.core.frame.DataFrame,imputed_values:pd.core.series.Series, nmf_model=NMF):
    '''
    predicts values for a new user based thier input dictionary
    needs the q_matrix to the used model and the imputed values 
    returns a DF for the user
    '''
    new_user_dataframe =  pd.DataFrame(data=user_query,
                columns=q_matrix.columns,
                index = ['new_user'])
    new_user_dataframe_imputed = new_user_dataframe.fillna(imputed_values)
    P_new_user_matrix = nmf_model.transform(new_user_dataframe_imputed)
    P_new_user = pd.DataFrame(data=P_new_user_matrix,
                            columns=nmf_model.get_feature_names_out(),
                            index = ['new_user'])
    R_hat_new_user_matrix = np.dot(P_new_user_matrix,q_matrix)
    R_hat_new_user = pd.DataFrame(data=R_hat_new_user_matrix,
                            columns=nmf_model.feature_names_in_,
                            index = ['new_user'])
    return R_hat_new_user

def get_top_rated_movies (new_user_r_matrix:pd.core.frame.DataFrame,n_top:int = 4):
    '''returns the top n number for the new user matrix'''
    return new_user_r_matrix.transpose().sort_values(by=['new_user'],ascending=False).sort_index().head(n_top)



#### Cossin neighbourghood

def get_cosine_similertiy(new_user_query:dict,original_DF:pd.core.frame.DataFrame):

    new_user_dataframe =  pd.DataFrame(data=new_user_query,
                columns=original_DF.columns,
                index = ['new_user'])

    dataframe_with_new_user = pd.concat([original_DF,new_user_dataframe],axis=0)
    dataframe_with_new_user =dataframe_with_new_user.T
    return dataframe_with_new_user, pd.DataFrame(cosine_similarity(dataframe_with_new_user.fillna(dataframe_with_new_user.mean()).T),columns = dataframe_with_new_user.columns, index = dataframe_with_new_user.columns)

     
def get_unseen_movies(data:pd.core.frame.DataFrame,user:str='new_user'):
    unseen = data[data['new_user'].isna()].index
    return unseen.tolist()

def get_similar_user(similiarity_matrix:pd.core.frame.DataFrame ,n:int):
    top_five_similar = similiarity_matrix['new_user'].sort_values(ascending= False).index[1:(n+1)]
    return top_five_similar.tolist()

def get_recommended_movies(unseen_movies:list,closes_users:list,data:pd.core.frame.DataFrame,similiarity_matrix:pd.core.frame.DataFrame,n_movies:int=5, user:str='new_user' ):
    movie_scores = pd.DataFrame()
    for movie in unseen_movies:
        others_user = data.columns[~data.loc[movie].isna()]
        others_user = set(others_user)
        if len(set(closes_users).intersection(others_user))>0:
            num = 0
            den = 0
            for user in set(closes_users).intersection(others_user): 
                ratings = data[user][movie] 
                sim = similiarity_matrix['new_user'][user]
                num = num + (ratings*sim)
                den = den + sim + 0.000001
        
            pred_ratings = num/den
            movie_scores = pd.concat([movie_scores,pd.DataFrame({'movie':[movie],'pred_ratings':[pred_ratings]})],axis=0)
    return movie_scores.sort_values(by='pred_ratings', ascending=False).head(n_movies)