from flask import Flask,render_template, request
from recommender import recommender_random, recommender_with_NMF, recommend_with_cossin
from utils import load_movies,load_data,get_initial_rating_df,load_NMF_and_imputed
app = Flask(__name__)
datapath_movies = "./data/ml-latest-small/movies.csv"
datapath_ratings = "./data/ml-latest-small/ratings.csv"
model_path = './models/model_mnf'
movie = load_movies(datapath_movies)
combined_df = load_data(ratings_path=datapath_ratings,movies_path=datapath_movies)
combined_df = get_initial_rating_df(combined_df)
model,imputed_values,Q_matrix,P_matrix  = load_NMF_and_imputed(pathfile=model_path)

@app.route('/')
def landing_page():
    '''
    renders landing page
    '''
    return render_template('index.html',name ='Max',movies = movie.title.to_list())

@app.route('/recommend')
def recommendations():
    titles = request.args.getlist('title')
    ratings = request.args.getlist('Ratings')
    user_input = dict(zip(titles,ratings))
    number_movies = int(request.args.getlist('number_movies')[0])
    print(number_movies)
    for keys in user_input: 
        user_input[keys] = int(user_input[keys])

    if request.args.get("Random") !='Random':
        recs = recommender_random(movie_list=movie['title'],n = number_movies) 
        return render_template('recommend.html',recs =recs)
    else: 
        if request.args['algorithm']=='NMF':
            recs = recommender_with_NMF(query=user_input,model=model,q_matrix=Q_matrix,imputed_values=imputed_values,movie_df=movie,n = number_movies)
            return render_template('recommend.html',recs =recs)
        elif request.args['algorithm']=='Cosine similarity':
            recs =  recommend_with_cossin(query=user_input,original_DF=combined_df,movie_df=movie,n = number_movies)
            return render_template('recommend.html',recs =recs)       
        else:
            return f"Function not defined"

if __name__=='__main__':
    app.run(debug=False,port=5000)
