import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import sys

# Neural Collaborative Filtering
from ncf_model import NCFModel, train_ncf_model

# Load datasets
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Basic checks
print("Movies shape:", movies.shape)
print("Ratings shape:", ratings.shape)

print("\nMovies sample:")
print(movies.head())

print("\nRatings sample:")
print(ratings.head())

print("\nMissing values:")
print(movies.isnull().sum())
print(ratings.isnull().sum())

print("\nUnique users:", ratings['userId'].nunique())
print("Unique movies:", ratings['movieId'].nunique())

# Split data for evaluation
train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)

# Create full matrix for SVD
user_movie_matrix = ratings.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
).fillna(0)

print("\nUser-Movie Matrix shape:")
print(user_movie_matrix.shape)

# Train SVD model
svd = TruncatedSVD(n_components=50, random_state=42)
user_factors = svd.fit_transform(user_movie_matrix)
movie_factors = svd.components_.T

# Reconstruct matrix
reconstructed = np.dot(user_factors, movie_factors.T)
reconstructed_df = pd.DataFrame(reconstructed, index=user_movie_matrix.index, columns=user_movie_matrix.columns)

# Evaluate on test set
test_predictions = []
test_actual = []
for _, row in test_ratings.iterrows():
    user = row['userId']
    movie = row['movieId']
    if user in reconstructed_df.index and movie in reconstructed_df.columns:
        pred = reconstructed_df.loc[user, movie]
        test_predictions.append(pred)
        test_actual.append(row['rating'])

rmse = np.sqrt(mean_squared_error(test_actual, test_predictions))
print(f"\nSVD Model RMSE on test set: {rmse:.4f}")

# Baseline: global mean
global_mean = train_ratings['rating'].mean()
baseline_predictions = [global_mean] * len(test_actual)
baseline_rmse = np.sqrt(mean_squared_error(test_actual, baseline_predictions))
print(f"Baseline RMSE (global mean): {baseline_rmse:.4f}")

# Compute cosine similarity between users
user_similarity = cosine_similarity(user_movie_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

def recommend_movies_collaborative(user_id, num_recommendations=5):
    if user_id not in user_similarity_df.index:
        return "User not found."
    
    # Get similar users
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:6]  # Top 5 similar
    
    # Get movies rated highly by similar users
    similar_users_ratings = user_movie_matrix.loc[similar_users]
    mean_ratings = similar_users_ratings.mean(axis=0)
    
    # Exclude movies already rated by the user
    user_rated = user_movie_matrix.loc[user_id][user_movie_matrix.loc[user_id] > 0].index
    recommendations = mean_ratings.drop(user_rated).sort_values(ascending=False).head(num_recommendations)
    
    # Get movie titles
    recommended_movies = movies[movies['movieId'].isin(recommendations.index)][['movieId', 'title']]
    recommended_movies['predicted_rating'] = recommendations.values
    
    return recommended_movies

# Popularity-based recommender
movie_stats = ratings.groupby('movieId').agg({'rating': ['mean', 'count']})
movie_stats.columns = ['avg_rating', 'num_ratings']
movie_stats = movie_stats.merge(movies[['movieId', 'title']], on='movieId')

def recommend_movies_popularity(user_id=None, num_recommendations=5, min_ratings=50):
    # Filter movies with minimum ratings
    popular_movies = movie_stats[movie_stats['num_ratings'] >= min_ratings].sort_values('avg_rating', ascending=False)
    
    if user_id and user_id in user_movie_matrix.index:
        # Exclude movies already rated by the user
        user_rated = user_movie_matrix.loc[user_id][user_movie_matrix.loc[user_id] > 0].index
        popular_movies = popular_movies[~popular_movies['movieId'].isin(user_rated)]
    
    return popular_movies[['movieId', 'title', 'avg_rating', 'num_ratings']].head(num_recommendations)

# Content-based recommender using genres
movies['genres_list'] = movies['genres'].str.split('|')

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def recommend_similar_movies(movie_id, num_recommendations=5):
    if movie_id not in movies['movieId'].values:
        return "Movie not found."
    
    movie_genres = set(movies[movies['movieId'] == movie_id]['genres_list'].values[0])
    movie_title = movies[movies['movieId'] == movie_id]['title'].values[0]
    
    # Compute similarities
    similarities = []
    for _, row in movies.iterrows():
        if row['movieId'] != movie_id:
            sim = jaccard_similarity(movie_genres, set(row['genres_list']))
            similarities.append((row['movieId'], row['title'], sim))
    
    # Sort by similarity descending
    similarities.sort(key=lambda x: x[2], reverse=True)
    
    # Get top recommendations
    recommended = similarities[:num_recommendations]
    recommended_df = pd.DataFrame(recommended, columns=['movieId', 'title', 'similarity'])
    
    return f"Similar movies to '{movie_title}' (ID: {movie_id}) based on genres:", recommended_df

# Mood-based recommender
mood_genres = {
    'happy': ['Comedy', 'Animation'],
    'sad': ['Drama'],
    'excited': ['Action', 'Adventure'],
    'relaxed': ['Romance', 'Comedy'],
    'scary': ['Thriller', 'Horror'],  # Note: Horror may not be present, but Thriller is
    'romantic': ['Romance'],
    'funny': ['Comedy']
}

def recommend_movies_by_mood(mood, num_recommendations=5, min_ratings=50):
    mood = mood.lower()
    if mood not in mood_genres:
        return "Mood not recognized. Available moods: " + ', '.join(mood_genres.keys())
    
    genres = mood_genres[mood]
    
    # Filter movies that have at least one of the genres
    mood_movies = movies[movies['genres'].str.contains('|'.join(genres), na=False)]
    
    # Merge with stats
    mood_stats = movie_stats[movie_stats['movieId'].isin(mood_movies['movieId'])]
    
    # Filter by min ratings
    mood_stats = mood_stats[mood_stats['num_ratings'] >= min_ratings]
    
    # Sort by average rating
    mood_stats = mood_stats.sort_values('avg_rating', ascending=False)
    
    return mood_stats[['movieId', 'title', 'avg_rating', 'num_ratings']].head(num_recommendations)

# Evaluation functions
def evaluate_precision_at_k(recommender_func, k=5, threshold=4.0):
    """
    Evaluate precision@K for a recommender function.
    For each user in test set, get K recommendations not seen in train,
    check how many have rating >= threshold in test.
    """
    precision_scores = []
    for user in test_ratings['userId'].unique()[:100]:  # Limit to 100 users for speed
        # Movies rated in train
        train_movies = train_ratings[train_ratings['userId'] == user]['movieId'].tolist()
        # Movies rated highly in test
        test_high = test_ratings[(test_ratings['userId'] == user) & (test_ratings['rating'] >= threshold)]['movieId'].tolist()
        
        if not test_high:
            continue  # No positive test ratings
        
        # Get recommendations
        recs = recommender_func(user, k)
        if isinstance(recs, str):
            continue  # User not found or error
        rec_movies = recs['movieId'].tolist()
        
        # Recommendations not in train
        new_recs = [m for m in rec_movies if m not in train_movies]
        
        # How many new recs are in test_high
        hits = len([m for m in new_recs if m in test_high])
        precision = hits / len(new_recs) if new_recs else 0
        precision_scores.append(precision)
    
    return np.mean(precision_scores) if precision_scores else 0

# Evaluate recommenders
print(f"\nCollaborative Filtering Precision@5: {evaluate_precision_at_k(recommend_movies_collaborative, k=5):.4f}")
print(f"Popularity-based Precision@5: {evaluate_precision_at_k(lambda u, k: recommend_movies_popularity(u, k), k=5):.4f}")

# Example recommendations
user_id = 1
print(f"\nCollaborative Recommendations for user {user_id}:")
print(recommend_movies_collaborative(user_id))

print(f"\nPopularity-based Recommendations for user {user_id}:")
print(recommend_movies_popularity(user_id))

print(f"\nTop Popular Movies (global):")
print(recommend_movies_popularity())

# ============================================
# NEURAL COLLABORATIVE FILTERING (Deep Learning)
# ============================================
print("\n" + "="*50)
print("TRAINING NEURAL COLLABORATIVE FILTERING MODEL")
print("="*50)

# Train NCF model (this may take a few minutes)
ncf_model, ncf_metrics = train_ncf_model(ratings, movies, embedding_dim=32, epochs=10, verbose=1)

# Compare all methods
print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)
print(f"{'Method':<35} {'RMSE':>10}")
print("-" * 45)
print(f"{'Baseline (Global Mean)':<35} {baseline_rmse:>10.4f}")
print(f"{'SVD Collaborative Filtering':<35} {rmse:>10.4f}")
print(f"{'Neural Collaborative Filtering':<35} {ncf_metrics['rmse']:>10.4f}")

# NCF Recommendations example
print(f"\nNCF Deep Learning Recommendations for user {user_id}:")
ncf_recommendations = ncf_model.recommend_movies(user_id, movies, ratings, n_recommendations=5)
print(ncf_recommendations)

# User input for mood-based recommendations
if len(sys.argv) > 1:
    mood_input = sys.argv[1]
else:
    try:
        mood_input = input("\nEnter your mood (happy, sad, excited, relaxed, scary, romantic, funny) to get movie recommendations: ")
    except EOFError:
        print("No mood provided. Run with: python main.py <mood>")
        sys.exit(1)

result = recommend_movies_by_mood(mood_input)
if isinstance(result, str):
    print(result)
else:
    print(f"\nRecommended movies for '{mood_input}' mood:")
    for _, row in result.iterrows():
        print(f"- {row['title']} (Avg Rating: {row['avg_rating']:.2f}, Ratings: {row['num_ratings']})")
