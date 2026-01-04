"""
Neural Collaborative Filtering (NCF) Model for Movie Recommendations

This module implements a deep learning-based recommender system using:
- User and Movie embeddings
- Multi-Layer Perceptron (MLP) for learning non-linear interactions
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


class NCFModel:
    """Neural Collaborative Filtering Model"""
    
    def __init__(self, num_users, num_movies, embedding_dim=32):
        """
        Initialize NCF Model
        
        Args:
            num_users: Number of unique users
            num_movies: Number of unique movies
            embedding_dim: Dimension of embedding vectors (default: 32)
        """
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_dim = embedding_dim
        self.model = self._build_model()
        self.user_encoder = None
        self.movie_encoder = None
        
    def _build_model(self):
        """Build the NCF neural network architecture"""
        
        # Input layers
        user_input = layers.Input(shape=(1,), name='user_input')
        movie_input = layers.Input(shape=(1,), name='movie_input')
        
        # User embedding
        user_embedding = layers.Embedding(
            input_dim=self.num_users,
            output_dim=self.embedding_dim,
            name='user_embedding'
        )(user_input)
        user_vec = layers.Flatten(name='user_flatten')(user_embedding)
        
        # Movie embedding
        movie_embedding = layers.Embedding(
            input_dim=self.num_movies,
            output_dim=self.embedding_dim,
            name='movie_embedding'
        )(movie_input)
        movie_vec = layers.Flatten(name='movie_flatten')(movie_embedding)
        
        # Concatenate user and movie vectors
        concat = layers.Concatenate(name='concat')([user_vec, movie_vec])
        
        # MLP layers with dropout for regularization
        x = layers.Dense(128, activation='relu', name='dense1')(concat)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu', name='dense2')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu', name='dense3')(x)
        
        # Output layer - predict rating (1-5 scale)
        output = layers.Dense(1, activation='linear', name='output')(x)
        
        # Build model
        model = Model(inputs=[user_input, movie_input], outputs=output, name='NCF')
        
        # Compile with Adam optimizer and MSE loss
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_data(self, ratings_df):
        """
        Prepare data for training
        
        Args:
            ratings_df: DataFrame with userId, movieId, rating columns
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Encode user and movie IDs to sequential integers
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        
        ratings_df = ratings_df.copy()
        ratings_df['user_encoded'] = self.user_encoder.fit_transform(ratings_df['userId'])
        ratings_df['movie_encoded'] = self.movie_encoder.fit_transform(ratings_df['movieId'])
        
        # Split data
        train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
        
        X_train = [train_df['user_encoded'].values, train_df['movie_encoded'].values]
        X_test = [test_df['user_encoded'].values, test_df['movie_encoded'].values]
        y_train = train_df['rating'].values
        y_test = test_df['rating'].values
        
        return X_train, X_test, y_train, y_test, train_df, test_df
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=20, batch_size=256, verbose=1):
        """
        Train the NCF model
        
        Args:
            X_train: Training features [user_ids, movie_ids]
            y_train: Training ratings
            X_val: Validation features (optional)
            y_val: Validation ratings (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.0001)
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history
    
    def predict(self, user_ids, movie_ids):
        """
        Predict ratings for user-movie pairs
        
        Args:
            user_ids: Array of user IDs (original)
            movie_ids: Array of movie IDs (original)
            
        Returns:
            Predicted ratings
        """
        # Encode IDs
        user_encoded = self.user_encoder.transform(user_ids)
        movie_encoded = self.movie_encoder.transform(movie_ids)
        
        predictions = self.model.predict([user_encoded, movie_encoded], verbose=0)
        return predictions.flatten()
    
    def recommend_movies(self, user_id, movies_df, ratings_df, n_recommendations=5):
        """
        Get top-N movie recommendations for a user
        
        Args:
            user_id: User ID to get recommendations for
            movies_df: DataFrame with movieId, title columns
            ratings_df: DataFrame with all ratings
            n_recommendations: Number of recommendations to return
            
        Returns:
            DataFrame with recommended movies
        """
        # Get movies the user hasn't rated
        user_rated = ratings_df[ratings_df['userId'] == user_id]['movieId'].values
        all_movies = movies_df['movieId'].values
        movies_to_predict = np.setdiff1d(all_movies, user_rated)
        
        # Filter to movies in our encoder
        known_movies = set(self.movie_encoder.classes_)
        movies_to_predict = [m for m in movies_to_predict if m in known_movies]
        
        if len(movies_to_predict) == 0:
            return pd.DataFrame(columns=['movieId', 'title', 'predicted_rating'])
        
        # Predict ratings for all unwatched movies
        user_ids = np.array([user_id] * len(movies_to_predict))
        movie_ids = np.array(movies_to_predict)
        
        try:
            predictions = self.predict(user_ids, movie_ids)
        except ValueError:
            return pd.DataFrame(columns=['movieId', 'title', 'predicted_rating'])
        
        # Create recommendations DataFrame
        recommendations = pd.DataFrame({
            'movieId': movies_to_predict,
            'predicted_rating': predictions
        })
        
        # Clip predictions to valid range
        recommendations['predicted_rating'] = recommendations['predicted_rating'].clip(1, 5)
        
        # Sort by predicted rating
        recommendations = recommendations.sort_values('predicted_rating', ascending=False)
        
        # Add movie titles
        recommendations = recommendations.merge(movies_df[['movieId', 'title']], on='movieId')
        
        return recommendations[['movieId', 'title', 'predicted_rating']].head(n_recommendations)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test data
        
        Args:
            X_test: Test features [user_ids, movie_ids]
            y_test: True ratings
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.model.predict(X_test, verbose=0).flatten()
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
        
        # Calculate MAE
        mae = np.mean(np.abs(predictions - y_test))
        
        return {
            'rmse': rmse,
            'mae': mae
        }
    
    def summary(self):
        """Print model summary"""
        return self.model.summary()


def train_ncf_model(ratings_df, movies_df, embedding_dim=32, epochs=15, verbose=1):
    """
    Convenience function to train an NCF model
    
    Args:
        ratings_df: Ratings DataFrame
        movies_df: Movies DataFrame
        embedding_dim: Embedding dimension
        epochs: Training epochs
        verbose: Verbosity level
        
    Returns:
        Trained NCFModel instance
    """
    num_users = ratings_df['userId'].nunique()
    num_movies = ratings_df['movieId'].nunique()
    
    print(f"Training NCF Model...")
    print(f"  Users: {num_users}, Movies: {num_movies}")
    print(f"  Embedding dimension: {embedding_dim}")
    
    # Create model
    ncf = NCFModel(num_users, num_movies, embedding_dim)
    
    # Prepare data
    X_train, X_test, y_train, y_test, _, _ = ncf.prepare_data(ratings_df)
    
    # Train
    ncf.train(X_train, y_train, X_test, y_test, epochs=epochs, verbose=verbose)
    
    # Evaluate
    metrics = ncf.evaluate(X_test, y_test)
    print(f"\nNCF Model Performance:")
    print(f"  Test RMSE: {metrics['rmse']:.4f}")
    print(f"  Test MAE: {metrics['mae']:.4f}")
    
    return ncf, metrics


# Standalone test
if __name__ == "__main__":
    # Load data
    print("Loading data...")
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    
    print(f"Loaded {len(ratings)} ratings for {ratings['movieId'].nunique()} movies")
    
    # Train model
    ncf_model, metrics = train_ncf_model(ratings, movies, epochs=10, verbose=1)
    
    # Example recommendation
    test_user = 1
    print(f"\nNCF Recommendations for User {test_user}:")
    recommendations = ncf_model.recommend_movies(test_user, movies, ratings, n_recommendations=5)
    print(recommendations)
