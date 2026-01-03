"""
Movie Recommender System - Flask Web Application
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import secrets
import requests
import re
warnings.filterwarnings('ignore')

# TMDB API Configuration
TMDB_API_KEY = '35bed244aad1623192dc64ce87945968'
TMDB_BASE_URL = 'https://api.themoviedb.org/3'
TMDB_IMAGE_BASE = 'https://image.tmdb.org/t/p/w342'

# Poster cache to avoid repeated API calls
poster_cache = {}

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Global data storage
movies = None
ratings = None
movie_stats = None
user_movie_matrix = None
user_similarity_df = None


def load_data():
    """Load movie and ratings data"""
    global movies, ratings, movie_stats, user_movie_matrix, user_similarity_df
    
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    
    # Compute movie stats
    movie_stats = ratings.groupby('movieId').agg({
        'rating': ['mean', 'count']
    }).reset_index()
    movie_stats.columns = ['movieId', 'avg_rating', 'num_ratings']
    
    # Build user-movie matrix
    user_movie_matrix = ratings.pivot_table(
        index='userId', columns='movieId', values='rating'
    ).fillna(0)
    
    # Compute user similarity
    user_similarity = cosine_similarity(user_movie_matrix)
    user_similarity_df = pd.DataFrame(
        user_similarity,
        index=user_movie_matrix.index,
        columns=user_movie_matrix.index
    )
    
    print(f"Loaded {len(movies)} movies and {len(ratings)} ratings")


def extract_year_from_title(title):
    """Extract year from movie title like 'Movie Name (1994)'"""
    match = re.search(r'\((\d{4})\)$', title)
    if match:
        return match.group(1)
    return None


def get_clean_title(title):
    """Remove year from movie title for search"""
    return re.sub(r'\s*\(\d{4}\)$', '', title).strip()


def get_poster_url(title, movie_id=None):
    """Fetch poster URL from TMDB API"""
    cache_key = movie_id if movie_id else title
    
    if cache_key in poster_cache:
        return poster_cache[cache_key]
    
    try:
        clean_title = get_clean_title(title)
        year = extract_year_from_title(title)
        
        params = {
            'api_key': TMDB_API_KEY,
            'query': clean_title,
            'language': 'en-US',
            'page': 1
        }
        
        if year:
            params['year'] = year
        
        response = requests.get(f'{TMDB_BASE_URL}/search/movie', params=params, timeout=3)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('results'):
                poster_path = data['results'][0].get('poster_path')
                if poster_path:
                    poster_url = f'{TMDB_IMAGE_BASE}{poster_path}'
                    poster_cache[cache_key] = poster_url
                    return poster_url
    except Exception:
        pass
    
    poster_cache[cache_key] = None
    return None


def enrich_movies_with_posters(movies_list):
    """Add poster URLs to a list of movie dictionaries"""
    for movie in movies_list:
        movie['poster_url'] = get_poster_url(movie.get('title', ''), movie.get('movieId'))
    return movies_list


def get_mood_genres(mood):
    """Map moods to genres"""
    mood_map = {
        'happy': ['Comedy', 'Animation'],
        'sad': ['Drama'],
        'excited': ['Action', 'Adventure'],
        'relaxed': ['Romance', 'Comedy'],
        'scary': ['Thriller', 'Horror'],
        'romantic': ['Romance'],
        'funny': ['Comedy'],
        'thoughtful': ['Documentary', 'Drama'],
        'family': ['Animation', 'Children', 'Family'],
        'nostalgic': ['Drama', 'Romance'],
        'adventurous': ['Adventure', 'Action', 'Fantasy'],
        'mysterious': ['Mystery', 'Thriller', 'Crime'],
        'inspiring': ['Drama', 'Documentary'],
        'chill': ['Comedy', 'Romance'],
        'dark': ['Crime', 'Thriller', 'Horror'],
        'heartwarming': ['Drama', 'Animation', 'Family'],
        'epic': ['Action', 'Adventure', 'War'],
        'mind-bending': ['Sci-Fi', 'Mystery', 'Thriller'],
        'feel-good': ['Comedy', 'Romance', 'Animation']
    }
    return mood_map.get(mood.lower(), ['Comedy'])


def recommend_by_mood(mood, n=12, min_ratings=30):
    """Get recommendations based on mood"""
    genres = get_mood_genres(mood)
    pattern = '|'.join(genres)
    
    mood_movies = movies[movies['genres'].str.contains(pattern, case=False, na=False)].copy()
    mood_movies = mood_movies.merge(movie_stats, on='movieId')
    mood_movies = mood_movies[mood_movies['num_ratings'] >= min_ratings]
    mood_movies = mood_movies.sort_values('avg_rating', ascending=False)
    
    movies_list = mood_movies.head(n).to_dict('records')
    return enrich_movies_with_posters(movies_list)


def recommend_by_genres(genres_list, n=12, min_ratings=30):
    """Get recommendations based on selected genres"""
    if not genres_list:
        return recommend_popular(n)
    
    pattern = '|'.join(genres_list)
    
    genre_movies = movies[movies['genres'].str.contains(pattern, case=False, na=False)].copy()
    genre_movies = genre_movies.merge(movie_stats, on='movieId')
    genre_movies = genre_movies[genre_movies['num_ratings'] >= min_ratings]
    genre_movies = genre_movies.sort_values('avg_rating', ascending=False)
    
    movies_list = genre_movies.head(n).to_dict('records')
    return enrich_movies_with_posters(movies_list)


def recommend_popular(n=12, min_ratings=50):
    """Popularity-based recommendations"""
    popular = movie_stats[movie_stats['num_ratings'] >= min_ratings].copy()
    popular = popular.sort_values('avg_rating', ascending=False).head(n)
    popular = popular.merge(movies[['movieId', 'title', 'genres']], on='movieId')
    movies_list = popular.to_dict('records')
    return enrich_movies_with_posters(movies_list)


def search_movies(query, n=20):
    """Search movies by title"""
    if not query:
        return []
    
    matches = movies[movies['title'].str.contains(query, case=False, na=False)].copy()
    matches = matches.merge(movie_stats, on='movieId', how='left')
    matches = matches.sort_values('avg_rating', ascending=False, na_position='last')
    movies_list = matches.head(n).to_dict('records')
    return enrich_movies_with_posters(movies_list)


# Available genres for signup
AVAILABLE_GENRES = [
    'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 
    'Documentary', 'Drama', 'Fantasy', 'Horror', 'Romance', 
    'Sci-Fi', 'Thriller'
]


# Routes
@app.route('/')
def home():
    """Home page with featured movies"""
    featured = recommend_popular(n=6, min_ratings=100)
    return render_template('index.html', 
                         featured=featured,
                         total_movies=len(movies),
                         total_ratings=len(ratings))


@app.route('/mood')
def mood_page():
    """Mood-based recommendations page"""
    mood = request.args.get('mood', 'happy')
    recommendations = recommend_by_mood(mood)
    moods = ['happy', 'sad', 'excited', 'relaxed', 'scary', 'romantic', 'funny', 'thoughtful', 'family',
             'nostalgic', 'adventurous', 'mysterious', 'inspiring', 'chill', 'dark', 'heartwarming', 'epic', 'mind-bending', 'feel-good']
    return render_template('mood.html', 
                         recommendations=recommendations, 
                         selected_mood=mood,
                         moods=moods)


@app.route('/signup', methods=['GET', 'POST'])
def signup_page():
    """User signup with genre interests"""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        interests = request.form.getlist('interests')
        
        if username and interests:
            session['user'] = {
                'username': username,
                'interests': interests
            }
            return redirect(url_for('profile_page'))
    
    return render_template('signup.html', genres=AVAILABLE_GENRES)


@app.route('/profile')
def profile_page():
    """User profile with personalized recommendations"""
    if 'user' not in session:
        return redirect(url_for('signup_page'))
    
    user = session['user']
    recommendations = recommend_by_genres(user['interests'], n=12)
    
    return render_template('profile.html',
                         user=user,
                         recommendations=recommendations)


@app.route('/logout')
def logout():
    """Clear user session"""
    session.clear()
    return redirect(url_for('home'))


@app.route('/search')
def search_page():
    """Search page"""
    query = request.args.get('q', '')
    results = search_movies(query) if query else []
    return render_template('search.html', results=results, query=query)


@app.route('/top-rated')
def top_rated_page():
    """Top rated movies page"""
    min_ratings = request.args.get('min_ratings', 50, type=int)
    top_movies = recommend_popular(n=12, min_ratings=min_ratings)
    return render_template('top_rated.html', 
                         movies=top_movies, 
                         min_ratings=min_ratings)


@app.route('/api/recommend/mood/<mood>')
def api_mood(mood):
    """API endpoint for mood-based recommendations"""
    recommendations = recommend_by_mood(mood)
    return jsonify(recommendations)


@app.route('/api/recommend/genres')
def api_genres():
    """API endpoint for genre-based recommendations"""
    genres = request.args.getlist('genre')
    recommendations = recommend_by_genres(genres)
    return jsonify(recommendations)


@app.route('/api/search')
def api_search():
    """API endpoint for search"""
    query = request.args.get('q', '')
    results = search_movies(query)
    return jsonify(results)


# Initialize data on startup
load_data()

if __name__ == '__main__':
    app.run(debug=True, port=5000)

