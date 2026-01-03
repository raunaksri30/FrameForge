"""
Movie Recommender System - Web Dashboard
Built with Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Cards */
    .movie-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
    }
    
    .movie-title {
        color: #e94560;
        font-size: 1.2em;
        font-weight: bold;
        margin-bottom: 8px;
    }
    
    .movie-rating {
        color: #ffd700;
        font-size: 1.1em;
    }
    
    .movie-genres {
        color: #a0a0a0;
        font-size: 0.9em;
        margin-top: 5px;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(26, 26, 46, 0.95);
    }
    
    /* Mood buttons */
    .mood-btn {
        background: linear-gradient(135deg, #e94560, #0f3460);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 25px;
        cursor: pointer;
        margin: 5px;
        font-weight: bold;
    }
    
    /* Stats cards */
    .stat-card {
        background: linear-gradient(135deg, #e94560, #533483);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: white;
    }
    
    .stat-number {
        font-size: 2.5em;
        font-weight: bold;
    }
    
    .stat-label {
        font-size: 0.9em;
        opacity: 0.9;
    }
    
    /* Search box */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        color: white;
        border-radius: 10px;
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and cache movie data"""
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    return movies, ratings


@st.cache_data
def compute_movie_stats(_ratings):
    """Compute movie statistics"""
    stats = _ratings.groupby('movieId').agg({
        'rating': ['mean', 'count']
    }).reset_index()
    stats.columns = ['movieId', 'avg_rating', 'num_ratings']
    return stats


@st.cache_data
def build_similarity_matrix(_ratings):
    """Build user similarity matrix"""
    user_movie_matrix = _ratings.pivot_table(
        index='userId', columns='movieId', values='rating'
    ).fillna(0)
    
    user_similarity = cosine_similarity(user_movie_matrix)
    user_similarity_df = pd.DataFrame(
        user_similarity,
        index=user_movie_matrix.index,
        columns=user_movie_matrix.index
    )
    return user_movie_matrix, user_similarity_df


def get_mood_genres(mood):
    """Map moods to genres"""
    mood_map = {
        '😊 Happy': ['Comedy', 'Animation'],
        '😢 Sad': ['Drama'],
        '🔥 Excited': ['Action', 'Adventure'],
        '😌 Relaxed': ['Romance', 'Comedy'],
        '😱 Scary': ['Thriller', 'Horror'],
        '💕 Romantic': ['Romance'],
        '😂 Funny': ['Comedy'],
        '🤔 Thoughtful': ['Documentary', 'Drama'],
        '👨‍👩‍👧‍👦 Family': ['Animation', 'Children', 'Family']
    }
    return mood_map.get(mood, ['Comedy'])


def recommend_by_mood(mood, movies, movie_stats, n=10, min_ratings=30):
    """Get recommendations based on mood"""
    genres = get_mood_genres(mood)
    
    # Filter movies with matching genres
    pattern = '|'.join(genres)
    mood_movies = movies[movies['genres'].str.contains(pattern, case=False, na=False)]
    
    # Merge with stats and filter
    mood_movies = mood_movies.merge(movie_stats, on='movieId')
    mood_movies = mood_movies[mood_movies['num_ratings'] >= min_ratings]
    
    # Sort by rating
    mood_movies = mood_movies.sort_values('avg_rating', ascending=False)
    
    return mood_movies.head(n)


def recommend_collaborative(user_id, user_movie_matrix, user_similarity_df, movies, n=10):
    """Collaborative filtering recommendations"""
    if user_id not in user_similarity_df.index:
        return pd.DataFrame()
    
    # Get similar users
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:11]
    
    # Get movies rated by similar users
    similar_ratings = user_movie_matrix.loc[similar_users]
    mean_ratings = similar_ratings.mean(axis=0)
    
    # Exclude already rated movies
    user_rated = user_movie_matrix.loc[user_id][user_movie_matrix.loc[user_id] > 0].index
    recommendations = mean_ratings.drop(user_rated, errors='ignore').sort_values(ascending=False).head(n)
    
    # Create DataFrame
    rec_df = movies[movies['movieId'].isin(recommendations.index)].copy()
    rec_df['predicted_rating'] = rec_df['movieId'].map(recommendations)
    rec_df = rec_df.sort_values('predicted_rating', ascending=False)
    
    return rec_df


def recommend_popular(movies, movie_stats, n=10, min_ratings=50):
    """Popularity-based recommendations"""
    popular = movie_stats[movie_stats['num_ratings'] >= min_ratings].copy()
    popular = popular.sort_values('avg_rating', ascending=False).head(n)
    popular = popular.merge(movies[['movieId', 'title', 'genres']], on='movieId')
    return popular


def search_movies(query, movies, movie_stats):
    """Search movies by title"""
    if not query:
        return pd.DataFrame()
    
    matches = movies[movies['title'].str.contains(query, case=False, na=False)].copy()
    matches = matches.merge(movie_stats, on='movieId', how='left')
    matches = matches.sort_values('avg_rating', ascending=False, na_position='last')
    return matches.head(20)


def display_movie_card(movie, show_predicted=False):
    """Display a movie as a styled card"""
    title = movie.get('title', 'Unknown')
    genres = movie.get('genres', 'N/A')
    avg_rating = movie.get('avg_rating', 0)
    num_ratings = movie.get('num_ratings', 0)
    predicted = movie.get('predicted_rating', 0)
    
    rating_stars = "⭐" * int(round(avg_rating))
    
    card_html = f"""
    <div class="movie-card">
        <div class="movie-title">🎬 {title}</div>
        <div class="movie-rating">{rating_stars} {avg_rating:.1f}/5 ({int(num_ratings)} ratings)</div>
        <div class="movie-genres">🎭 {genres}</div>
        {f'<div style="color: #4ade80; margin-top: 8px;">📊 Predicted: {predicted:.2f}</div>' if show_predicted and predicted > 0 else ''}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def main():
    # Load data
    movies, ratings = load_data()
    movie_stats = compute_movie_stats(ratings)
    user_movie_matrix, user_similarity_df = build_similarity_matrix(ratings)
    
    # Sidebar
    with st.sidebar:
        st.markdown("# 🎬 Movie Recommender")
        st.markdown("---")
        
        # Stats
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Movies", f"{len(movies):,}")
        with col2:
            st.metric("Ratings", f"{len(ratings):,}")
        
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigate",
            ["🏠 Home", "🎭 By Mood", "👤 For You", "🔍 Search", "🏆 Top Rated"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This recommender uses:
        - 📊 Collaborative Filtering
        - 🧠 Neural Networks (NCF)
        - 🎭 Content-based Filtering
        - ⭐ Popularity Ranking
        """)
    
    # Main content
    if page == "🏠 Home":
        st.title("🎬 Welcome to Movie Recommender")
        st.markdown("### Discover your next favorite movie!")
        
        # Featured section
        st.markdown("---")
        st.subheader("🌟 Featured Movies")
        
        featured = recommend_popular(movies, movie_stats, n=6, min_ratings=100)
        
        cols = st.columns(3)
        for idx, (_, movie) in enumerate(featured.iterrows()):
            with cols[idx % 3]:
                display_movie_card(movie)
        
        # Quick moods
        st.markdown("---")
        st.subheader("🎭 Quick Mood Selection")
        
        mood_cols = st.columns(5)
        moods = ['😊 Happy', '🔥 Excited', '💕 Romantic', '😱 Scary', '😂 Funny']
        
        for idx, mood in enumerate(moods):
            with mood_cols[idx]:
                if st.button(mood, key=f"home_mood_{idx}", use_container_width=True):
                    st.session_state['selected_mood'] = mood
                    st.rerun()
    
    elif page == "🎭 By Mood":
        st.title("🎭 Mood-Based Recommendations")
        st.markdown("### How are you feeling today?")
        
        moods = ['😊 Happy', '😢 Sad', '🔥 Excited', '😌 Relaxed', '😱 Scary', 
                 '💕 Romantic', '😂 Funny', '🤔 Thoughtful', '👨‍👩‍👧‍👦 Family']
        
        selected_mood = st.selectbox("Select your mood:", moods)
        
        if selected_mood:
            st.markdown(f"### Movies for {selected_mood} mood:")
            
            recommendations = recommend_by_mood(selected_mood, movies, movie_stats, n=9)
            
            if len(recommendations) > 0:
                cols = st.columns(3)
                for idx, (_, movie) in enumerate(recommendations.iterrows()):
                    with cols[idx % 3]:
                        display_movie_card(movie)
            else:
                st.warning("No movies found for this mood. Try another!")
    
    elif page == "👤 For You":
        st.title("👤 Personalized Recommendations")
        st.markdown("### Get recommendations based on your taste!")
        
        # User selection
        available_users = sorted(ratings['userId'].unique())
        selected_user = st.selectbox(
            "Select your User ID:", 
            available_users,
            help="Choose a user ID to get personalized recommendations"
        )
        
        if selected_user:
            # Show user stats
            user_ratings = ratings[ratings['userId'] == selected_user]
            user_avg = user_ratings['rating'].mean()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Movies Rated", len(user_ratings))
            with col2:
                st.metric("Average Rating", f"{user_avg:.1f} ⭐")
            with col3:
                st.metric("User ID", selected_user)
            
            st.markdown("---")
            st.subheader("🎯 Recommended for You")
            
            recommendations = recommend_collaborative(
                selected_user, user_movie_matrix, user_similarity_df, movies, n=9
            )
            recommendations = recommendations.merge(movie_stats, on='movieId', how='left')
            
            if len(recommendations) > 0:
                cols = st.columns(3)
                for idx, (_, movie) in enumerate(recommendations.iterrows()):
                    with cols[idx % 3]:
                        display_movie_card(movie, show_predicted=True)
            else:
                st.info("Not enough data to make recommendations for this user.")
    
    elif page == "🔍 Search":
        st.title("🔍 Search Movies")
        
        query = st.text_input("Search for a movie:", placeholder="Enter movie title...")
        
        if query:
            results = search_movies(query, movies, movie_stats)
            
            if len(results) > 0:
                st.markdown(f"### Found {len(results)} movies matching '{query}'")
                
                cols = st.columns(3)
                for idx, (_, movie) in enumerate(results.iterrows()):
                    with cols[idx % 3]:
                        display_movie_card(movie)
            else:
                st.warning(f"No movies found matching '{query}'")
        else:
            st.info("Start typing to search for movies...")
    
    elif page == "🏆 Top Rated":
        st.title("🏆 Top Rated Movies")
        
        min_ratings = st.slider("Minimum number of ratings:", 10, 200, 50)
        
        top_movies = recommend_popular(movies, movie_stats, n=12, min_ratings=min_ratings)
        
        if len(top_movies) > 0:
            cols = st.columns(3)
            for idx, (_, movie) in enumerate(top_movies.iterrows()):
                with cols[idx % 3]:
                    display_movie_card(movie)
        else:
            st.warning("No movies found with that many ratings. Try lowering the minimum.")


if __name__ == "__main__":
    main()
