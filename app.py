"""
Movie Recommender System - Flask Web Application
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import secrets
import requests
import re
import os
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

# Load environment variables from .env file
basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

# TMDB API Configuration - Set TMDB_API_KEY in .env file
TMDB_API_KEY = os.environ.get('TMDB_API_KEY', '')
TMDB_BASE_URL = 'https://api.themoviedb.org/3'
TMDB_IMAGE_BASE = 'https://image.tmdb.org/t/p/w342'

# Poster cache to avoid repeated API calls
poster_cache = {}

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Database Configuration
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    interests = db.Column(db.String(500), nullable=True)  # Comma-separated genres
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def get_interests_list(self):
        if self.interests:
            return self.interests.split(',')
        return []
    
    def set_interests_list(self, interests_list):
        self.interests = ','.join(interests_list)


# Create database tables
with app.app_context():
    db.create_all()


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


# Cache for movie details to avoid repeated API calls
details_cache = {}


def get_movie_details(title, movie_id=None):
    """Fetch detailed movie info from TMDB API (overview, cast, rating, tmdb_id)"""
    cache_key = movie_id if movie_id else title
    
    if cache_key in details_cache:
        return details_cache[cache_key]
    
    try:
        clean_title = get_clean_title(title)
        year = extract_year_from_title(title)
        
        # Search for the movie
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
                movie = data['results'][0]
                tmdb_id = movie.get('id')
                
                details = {
                    'overview': movie.get('overview', ''),
                    'tmdb_rating': movie.get('vote_average', 0),
                    'tmdb_id': tmdb_id,
                    'cast': []
                }
                
                # Fetch cast information
                if tmdb_id:
                    try:
                        credits_response = requests.get(
                            f'{TMDB_BASE_URL}/movie/{tmdb_id}/credits',
                            params={'api_key': TMDB_API_KEY},
                            timeout=3
                        )
                        if credits_response.status_code == 200:
                            credits_data = credits_response.json()
                            cast_list = credits_data.get('cast', [])[:5]  # Top 5 actors
                            details['cast'] = [actor.get('name', '') for actor in cast_list]
                    except Exception:
                        pass
                
                details_cache[cache_key] = details
                return details
    except Exception:
        pass
    
    details_cache[cache_key] = {'overview': '', 'tmdb_rating': 0, 'cast': [], 'tmdb_id': None}
    return details_cache[cache_key]


def enrich_movies_with_posters(movies_list):
    """Add poster URLs and movie details to a list of movie dictionaries"""
    for movie in movies_list:
        movie['poster_url'] = get_poster_url(movie.get('title', ''), movie.get('movieId'))
        # Fetch additional movie details (overview, cast, tmdb_rating, tmdb_id)
        details = get_movie_details(movie.get('title', ''), movie.get('movieId'))
        movie['overview'] = details.get('overview', '')
        movie['cast'] = details.get('cast', [])
        movie['tmdb_rating'] = details.get('tmdb_rating', 0)
        movie['tmdb_id'] = details.get('tmdb_id')
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


def recommend_by_mood(moods, n=12, min_ratings=30):
    """Get recommendations based on one or more moods"""
    if isinstance(moods, str):
        moods = [moods]
    
    # Collect all genres from all selected moods
    all_genres = []
    for mood in moods:
        all_genres.extend(get_mood_genres(mood))
    
    # Remove duplicates while preserving order
    unique_genres = list(dict.fromkeys(all_genres))
    pattern = '|'.join(unique_genres)
    
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
    """Home page with categorized movie sections"""
    # Trending/Popular movies
    trending = recommend_popular(n=6, min_ratings=100)
    
    # Genre-based sections
    top_action = recommend_by_genres(['Action', 'Adventure'], n=6, min_ratings=30)
    top_drama = recommend_by_genres(['Drama'], n=6, min_ratings=30)
    top_comedy = recommend_by_genres(['Comedy'], n=6, min_ratings=30)
    top_thriller = recommend_by_genres(['Thriller', 'Horror'], n=6, min_ratings=30)
    
    # Personalized "New For You" if user is logged in
    new_for_you = []
    if 'user' in session and session['user'].get('interests'):
        new_for_you = recommend_by_genres(session['user']['interests'], n=6, min_ratings=20)
    
    return render_template('index.html', 
                         trending=trending,
                         new_for_you=new_for_you,
                         top_action=top_action,
                         top_drama=top_drama,
                         top_comedy=top_comedy,
                         top_thriller=top_thriller,
                         total_movies=len(movies),
                         total_ratings=len(ratings))


@app.route('/mood')
def mood_page():
    """Mood-based recommendations page"""
    selected_moods = request.args.getlist('mood')
    moods = ['happy', 'sad', 'excited', 'relaxed', 'scary', 'romantic', 'funny', 'thoughtful', 'family',
             'nostalgic', 'adventurous', 'mysterious', 'inspiring', 'chill', 'dark', 'heartwarming', 'epic', 'mind-bending', 'feel-good']
    
    recommendations = recommend_by_mood(selected_moods) if selected_moods else []
    
    return render_template('mood.html', 
                         recommendations=recommendations, 
                         selected_moods=selected_moods,
                         moods=moods)


@app.route('/signup', methods=['GET', 'POST'])
def signup_page():
    """User signup with email, password and genre interests"""
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        username = request.form.get('username', '').strip()
        interests = request.form.getlist('interests')
        
        if email and password and username:
            # Check if user already exists
            existing_user = User.query.filter(
                (User.email == email) | (User.username == username)
            ).first()
            
            if existing_user:
                flash('Username or email already exists!', 'error')
                return render_template('signup.html', genres=AVAILABLE_GENRES)
            
            # Create new user
            new_user = User(username=username, email=email)
            new_user.set_password(password)
            new_user.set_interests_list(interests)
            
            db.session.add(new_user)
            db.session.commit()
            
            # Log user in
            session['user_id'] = new_user.id
            flash('Account created successfully!', 'success')
            return redirect(url_for('profile_page'))
    
    return render_template('signup.html', genres=AVAILABLE_GENRES)


@app.route('/login', methods=['GET', 'POST'])
def login_page():
    """User login page"""
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        
        if email and password:
            user = User.query.filter_by(email=email).first()
            
            if user and user.check_password(password):
                session['user_id'] = user.id
                flash('Logged in successfully!', 'success')
                return redirect(url_for('profile_page'))
            else:
                flash('Invalid email or password!', 'error')
    
    return render_template('login.html')


@app.route('/profile')
def profile_page():
    """User profile with personalized recommendations"""
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    
    user = User.query.get(session['user_id'])
    if not user:
        session.clear()
        return redirect(url_for('login_page'))
    
    # Get recommendations based on user interests
    interests = user.get_interests_list()
    recommendations = recommend_by_genres(interests, n=12) if interests else recommend_popular(n=12)
    
    # Create user dict for template compatibility
    user_data = {
        'username': user.username,
        'email': user.email,
        'interests': interests
    }
    
    return render_template('profile.html',
                         user=user_data,
                         recommendations=recommendations)


@app.route('/logout')
def logout():
    """Clear user session"""
    session.clear()
    flash('Logged out successfully!', 'success')
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
    top_movies = recommend_popular(n=24, min_ratings=min_ratings)
    return render_template('top_rated.html', 
                         movies=top_movies, 
                         min_ratings=min_ratings,
                         page_title='Trending Now')


@app.route('/browse/<category>')
def browse_category(category):
    """Browse movies by category"""
    category_config = {
        'action': {
            'title': 'Action & Adventure',
            'genres': ['Action', 'Adventure']
        },
        'drama': {
            'title': 'Drama',
            'genres': ['Drama']
        },
        'comedy': {
            'title': 'Comedy',
            'genres': ['Comedy']
        },
        'thriller': {
            'title': 'Thriller & Horror',
            'genres': ['Thriller', 'Horror']
        },
        'romance': {
            'title': 'Romance',
            'genres': ['Romance']
        },
        'scifi': {
            'title': 'Sci-Fi & Fantasy',
            'genres': ['Sci-Fi', 'Fantasy']
        }
    }
    
    config = category_config.get(category, {'title': 'Movies', 'genres': []})
    min_ratings = request.args.get('min_ratings', 20, type=int)
    movies_list = recommend_by_genres(config['genres'], n=24, min_ratings=min_ratings)
    
    return render_template('browse.html',
                         movies=movies_list,
                         category_title=config['title'],
                         min_ratings=min_ratings,
                         category=category)


@app.route('/watch/<int:movie_id>')
def watch_page(movie_id):
    """Watch movie page with embedded Vidking player"""
    # Find movie by ID
    movie_row = movies[movies['movieId'] == movie_id]
    
    if movie_row.empty:
        return redirect(url_for('home'))
    
    movie_data = movie_row.iloc[0].to_dict()
    
    # Get movie stats
    stats = movie_stats[movie_stats['movieId'] == movie_id]
    if not stats.empty:
        movie_data['avg_rating'] = stats.iloc[0]['avg_rating']
        movie_data['num_ratings'] = stats.iloc[0]['num_ratings']
    else:
        movie_data['avg_rating'] = 0
        movie_data['num_ratings'] = 0
    
    # Enrich with poster and details
    enriched = enrich_movies_with_posters([movie_data])[0]
    
    # Get similar movies for recommendations
    similar_genres = enriched.get('genres', '').split('|')[:2]
    similar_movies = recommend_by_genres(similar_genres, n=6, min_ratings=20)
    # Filter out the current movie
    similar_movies = [m for m in similar_movies if m.get('movieId') != movie_id][:5]
    
    return render_template('watch.html',
                         movie=enriched,
                         similar_movies=similar_movies)


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
    app.run(host='0.0.0.0', debug=False, port=5000)

