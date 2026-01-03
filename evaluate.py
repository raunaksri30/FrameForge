"""
Comprehensive Evaluation Script for Movie Recommender System

Metrics included:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- Precision@K
- Recall@K
- NDCG@K (Normalized Discounted Cumulative Gain)
- Coverage (catalog coverage)
- Diversity (how varied recommendations are)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class RecommenderEvaluator:
    """Comprehensive evaluator for recommender systems"""
    
    def __init__(self, ratings_df, movies_df, test_size=0.2, random_state=42):
        """
        Initialize evaluator with data
        
        Args:
            ratings_df: DataFrame with userId, movieId, rating
            movies_df: DataFrame with movieId, title, genres
            test_size: Fraction of data for testing
            random_state: Random seed for reproducibility
        """
        self.ratings = ratings_df
        self.movies = movies_df
        self.train, self.test = train_test_split(
            ratings_df, test_size=test_size, random_state=random_state
        )
        
        # Create user-item matrix from training data
        self.train_matrix = self.train.pivot_table(
            index='userId', columns='movieId', values='rating'
        ).fillna(0)
        
        # Get all unique items
        self.all_items = set(movies_df['movieId'].values)
        
        # Relevance threshold (ratings >= this are considered "liked")
        self.relevance_threshold = 4.0
        
        print(f"Evaluator initialized:")
        print(f"  Training samples: {len(self.train)}")
        print(f"  Test samples: {len(self.test)}")
        print(f"  Users: {self.train['userId'].nunique()}")
        print(f"  Movies: {self.train['movieId'].nunique()}")
    
    def rmse(self, predictions, actuals):
        """Calculate Root Mean Squared Error"""
        return np.sqrt(mean_squared_error(actuals, predictions))
    
    def mae(self, predictions, actuals):
        """Calculate Mean Absolute Error"""
        return mean_absolute_error(actuals, predictions)
    
    def precision_at_k(self, recommended_items, relevant_items, k):
        """
        Precision@K: What fraction of top-K recommendations are relevant?
        
        Args:
            recommended_items: List of recommended item IDs (ordered by score)
            relevant_items: Set of actually relevant item IDs
            k: Number of top recommendations to consider
        """
        if k == 0:
            return 0.0
        
        top_k = recommended_items[:k]
        hits = len(set(top_k) & set(relevant_items))
        return hits / k
    
    def recall_at_k(self, recommended_items, relevant_items, k):
        """
        Recall@K: What fraction of relevant items are in top-K?
        
        Args:
            recommended_items: List of recommended item IDs
            relevant_items: Set of actually relevant item IDs
            k: Number of top recommendations to consider
        """
        if len(relevant_items) == 0:
            return 0.0
        
        top_k = recommended_items[:k]
        hits = len(set(top_k) & set(relevant_items))
        return hits / len(relevant_items)
    
    def dcg_at_k(self, scores, k):
        """Discounted Cumulative Gain at K"""
        scores = np.array(scores)[:k]
        if len(scores) == 0:
            return 0.0
        
        # DCG = sum of (relevance / log2(position + 1))
        positions = np.arange(1, len(scores) + 1)
        return np.sum(scores / np.log2(positions + 1))
    
    def ndcg_at_k(self, recommended_items, relevant_items, k):
        """
        Normalized Discounted Cumulative Gain at K
        
        Measures ranking quality - higher scores if relevant items appear earlier
        """
        # Binary relevance scores for recommended items
        relevance = [1 if item in relevant_items else 0 for item in recommended_items[:k]]
        
        # Calculate DCG
        dcg = self.dcg_at_k(relevance, k)
        
        # Calculate ideal DCG (all relevant items at top)
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = self.dcg_at_k(ideal_relevance, k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def catalog_coverage(self, all_recommendations):
        """
        Coverage: What fraction of catalog is ever recommended?
        
        Args:
            all_recommendations: List of all recommended items across all users
        """
        unique_recommended = set(all_recommendations)
        return len(unique_recommended) / len(self.all_items)
    
    def diversity(self, recommended_items, movies_df):
        """
        Diversity: How varied are the genres in recommendations?
        
        Higher diversity = recommendations span more genres
        """
        if len(recommended_items) <= 1:
            return 0.0
        
        # Get genres for recommended items
        rec_movies = movies_df[movies_df['movieId'].isin(recommended_items)]
        
        if len(rec_movies) == 0:
            return 0.0
        
        # Calculate pairwise genre dissimilarity
        genre_sets = []
        for genres in rec_movies['genres'].fillna(''):
            genre_sets.append(set(genres.split('|')))
        
        # Average Jaccard distance between all pairs
        distances = []
        for i in range(len(genre_sets)):
            for j in range(i + 1, len(genre_sets)):
                intersection = len(genre_sets[i] & genre_sets[j])
                union = len(genre_sets[i] | genre_sets[j])
                if union > 0:
                    jaccard_sim = intersection / union
                    distances.append(1 - jaccard_sim)  # Distance = 1 - similarity
        
        return np.mean(distances) if distances else 0.0
    
    def evaluate_recommender(self, recommend_func, k=10, num_users=100, name="Recommender"):
        """
        Comprehensive evaluation of a recommender function
        
        Args:
            recommend_func: Function that takes (user_id, k) and returns list of movieIds
            k: Number of recommendations to evaluate
            num_users: Number of users to evaluate (for speed)
            name: Name of the recommender for display
            
        Returns:
            Dictionary of metrics
        """
        print(f"\nEvaluating: {name}")
        print("-" * 40)
        
        precisions = []
        recalls = []
        ndcgs = []
        diversities = []
        all_recommendations = []
        
        # Get test users with enough ratings
        test_users = self.test['userId'].unique()[:num_users]
        
        for user_id in test_users:
            # Get items the user rated highly in test set
            user_test = self.test[self.test['userId'] == user_id]
            relevant_items = set(
                user_test[user_test['rating'] >= self.relevance_threshold]['movieId'].values
            )
            
            if len(relevant_items) == 0:
                continue
            
            # Get recommendations
            try:
                recommendations = recommend_func(user_id, k)
                if recommendations is None or len(recommendations) == 0:
                    continue
                
                # Handle both list and DataFrame returns
                if isinstance(recommendations, pd.DataFrame):
                    rec_items = recommendations['movieId'].tolist()
                else:
                    rec_items = list(recommendations)
                
            except Exception:
                continue
            
            # Calculate metrics
            precisions.append(self.precision_at_k(rec_items, relevant_items, k))
            recalls.append(self.recall_at_k(rec_items, relevant_items, k))
            ndcgs.append(self.ndcg_at_k(rec_items, relevant_items, k))
            diversities.append(self.diversity(rec_items, self.movies))
            all_recommendations.extend(rec_items)
        
        # Aggregate metrics
        metrics = {
            'precision@k': np.mean(precisions) if precisions else 0,
            'recall@k': np.mean(recalls) if recalls else 0,
            'ndcg@k': np.mean(ndcgs) if ndcgs else 0,
            'diversity': np.mean(diversities) if diversities else 0,
            'coverage': self.catalog_coverage(all_recommendations),
            'users_evaluated': len(precisions)
        }
        
        # Print results
        print(f"  Precision@{k}:  {metrics['precision@k']:.4f}")
        print(f"  Recall@{k}:     {metrics['recall@k']:.4f}")
        print(f"  NDCG@{k}:       {metrics['ndcg@k']:.4f}")
        print(f"  Diversity:      {metrics['diversity']:.4f}")
        print(f"  Coverage:       {metrics['coverage']:.4f} ({int(metrics['coverage'] * len(self.all_items))}/{len(self.all_items)} items)")
        print(f"  Users tested:   {metrics['users_evaluated']}")
        
        return metrics


def run_full_evaluation():
    """Run comprehensive evaluation on all recommenders"""
    
    print("=" * 60)
    print("COMPREHENSIVE RECOMMENDER SYSTEM EVALUATION")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    
    # Initialize evaluator
    evaluator = RecommenderEvaluator(ratings, movies)
    
    # Import recommenders from main
    print("\nLoading recommenders...")
    
    # Build user-movie matrix for collaborative filtering
    from sklearn.metrics.pairwise import cosine_similarity
    
    user_movie_matrix = ratings.pivot_table(
        index='userId', columns='movieId', values='rating'
    ).fillna(0)
    
    user_similarity = cosine_similarity(user_movie_matrix)
    user_similarity_df = pd.DataFrame(
        user_similarity, 
        index=user_movie_matrix.index, 
        columns=user_movie_matrix.index
    )
    
    # Movie stats for popularity-based
    movie_stats = ratings.groupby('movieId').agg({'rating': ['mean', 'count']})
    movie_stats.columns = ['avg_rating', 'num_ratings']
    movie_stats = movie_stats.reset_index()
    movie_stats = movie_stats.merge(movies[['movieId', 'title']], on='movieId')
    
    # Define recommender functions
    def collaborative_recommend(user_id, k):
        if user_id not in user_similarity_df.index:
            return pd.DataFrame(columns=['movieId'])
        
        similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:6]
        similar_users_ratings = user_movie_matrix.loc[similar_users]
        mean_ratings = similar_users_ratings.mean(axis=0)
        
        user_rated = user_movie_matrix.loc[user_id][user_movie_matrix.loc[user_id] > 0].index
        recommendations = mean_ratings.drop(user_rated, errors='ignore').sort_values(ascending=False).head(k)
        
        return pd.DataFrame({'movieId': recommendations.index})
    
    def popularity_recommend(user_id, k):
        popular = movie_stats[movie_stats['num_ratings'] >= 50].sort_values('avg_rating', ascending=False)
        
        if user_id in user_movie_matrix.index:
            user_rated = user_movie_matrix.loc[user_id][user_movie_matrix.loc[user_id] > 0].index
            popular = popular[~popular['movieId'].isin(user_rated)]
        
        return popular[['movieId']].head(k)
    
    # Evaluate traditional methods
    results = {}
    
    results['Collaborative Filtering'] = evaluator.evaluate_recommender(
        collaborative_recommend, k=10, num_users=100, name="Collaborative Filtering"
    )
    
    results['Popularity-based'] = evaluator.evaluate_recommender(
        popularity_recommend, k=10, num_users=100, name="Popularity-based"
    )
    
    # Try to evaluate NCF if available
    try:
        from ncf_model import train_ncf_model
        print("\nTraining NCF model for evaluation...")
        ncf_model, _ = train_ncf_model(ratings, movies, epochs=5, verbose=0)
        
        def ncf_recommend(user_id, k):
            return ncf_model.recommend_movies(user_id, movies, ratings, n_recommendations=k)
        
        results['Neural CF (NCF)'] = evaluator.evaluate_recommender(
            ncf_recommend, k=10, num_users=100, name="Neural Collaborative Filtering"
        )
    except Exception as e:
        print(f"\nNote: Could not evaluate NCF model: {e}")
    
    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Method':<25} {'Prec@10':>10} {'Recall@10':>10} {'NDCG@10':>10} {'Coverage':>10}")
    print("-" * 65)
    
    for name, metrics in results.items():
        print(f"{name:<25} {metrics['precision@k']:>10.4f} {metrics['recall@k']:>10.4f} {metrics['ndcg@k']:>10.4f} {metrics['coverage']:>10.4f}")
    
    print("\n" + "=" * 60)
    print("METRIC EXPLANATIONS")
    print("=" * 60)
    print("""
  • Precision@K: Fraction of recommendations that are relevant
  • Recall@K: Fraction of relevant items that are recommended  
  • NDCG@K: Ranking quality (rewards relevant items appearing earlier)
  • Diversity: How varied the recommended genres are
  • Coverage: Fraction of catalog ever recommended
    """)
    
    return results


if __name__ == "__main__":
    run_full_evaluation()
