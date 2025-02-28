import streamlit as st
import pickle
import pandas as pd
import requests

# Function to fetch the movie poster using the TMDB API
def fetch_poster(movie_id):
    response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=c4c9897af1ecb7aa91be175d81e5694f&language=en-US')
    data = response.json()
    return f"https://image.tmdb.org/t/p/w185/{data['poster_path']}"

# Function to fetch the movie rating using the TMDB API
def fetch_rating(movie_id):
    response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=c4c9897af1ecb7aa91be175d81e5694f&language=en-US')
    data = response.json()
    return data['vote_average']

# Loading the precomputed movies data and similarity matrix
movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

# Recommendation function
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_movies_posters = []
    movie_ratings = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_movies_posters.append(fetch_poster(movie_id))
        movie_ratings.append(fetch_rating(movie_id))
    
    return recommended_movies, recommended_movies_posters, movie_ratings

# Streamlit UI
st.title(' :rainbow[Movie Recommender System]')
st.markdown('## Find Your Next Favorite Movie')

# Movie selection dropdown
selected_movie_name = st.selectbox(
   "Select your favorite movie",
   movies['title'].values
)

# Recommendation button
if st.button('Recommend'):
    names, posters, ratings = recommend(selected_movie_name)
    st.write('### Movies you may also like:')

    # Displaying recommended movies in columns
    cols = st.columns(5)
    for col, name, poster, rating in zip(cols, names, posters, ratings):
        with col:
            st.image(poster, use_container_width=True)
            st.markdown(f"**{name}**")
            st.markdown(f"Rating: {rating}")

# Sidebar with improved UI
st.sidebar.markdown("<h2 style='text-align: center;'>About</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p >This app recommends movies based on your favorite movie. It uses content-based filtering to suggest similar movies.</p>", unsafe_allow_html=True)

st.sidebar.markdown("<h2 style='text-align: center;'>Contact</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p>Developed by <b>Shaik Yousuf</b>. I am a passionate data scientist and machine learning enthusiast. Feel free to connect with me on LinkedIn.</p>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center;'><a href='https://www.linkedin.com/in/shaik-yousuf-a39566228/' target='_blank'><img src='https://img.icons8.com/fluency/48/000000/linkedin.png' alt='LinkedIn'></a></p>", unsafe_allow_html=True)

st.sidebar.markdown("<h2>Movie Recommender System</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p>Version 1.0</p>", unsafe_allow_html=True)
