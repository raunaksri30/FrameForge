# 🎬 FrameForge - Movie Recommender System

A full-stack movie recommendation system with multiple recommendation algorithms, a beautiful Flask web interface, and real movie posters from TMDB.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)

---

## 📋 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Web Application](#-web-application)
- [API Endpoints](#-api-endpoints)
- [Recommendation Algorithms](#-recommendation-algorithms)
- [Dataset](#-dataset)
- [Evaluation Metrics](#-evaluation-metrics)
- [Configuration](#-configuration)

---

## ✨ Features

### Web Application
- 🎭 **Mood-Based Discovery** - Find movies based on your current mood (happy, sad, excited, romantic, etc.)
- ⭐ **Top Rated Movies** - Browse the highest-rated movies with customizable rating thresholds
- 🔍 **Smart Search** - Search through 9,700+ movies by title
- 👤 **User Profiles** - Sign up and get personalized recommendations based on your genre preferences
- 🖼️ **Real Movie Posters** - High-quality posters fetched from TMDB API
- 📺 **Streaming Links** - Quick links to Google and JustWatch to find where to watch

### Recommendation Algorithms
- **Collaborative Filtering** - User-based recommendations using cosine similarity
- **Matrix Factorization (SVD)** - Latent factor model using TruncatedSVD
- **Neural Collaborative Filtering (NCF)** - Deep learning model with user/movie embeddings
- **Content-Based Filtering** - Genre similarity using Jaccard similarity
- **Popularity-Based** - Top-rated movies filtered by minimum ratings
- **Mood-Based** - Genre mapping for mood-specific recommendations

---

## 📁 Project Structure

```
movie recommender/
├── flask_app.py          # Flask web application (main entry point)
├── main.py               # Core recommendation logic & algorithms
├── ncf_model.py          # Neural Collaborative Filtering (Deep Learning)
├── evaluate.py           # Comprehensive evaluation metrics
├── app.py                # Alternative Streamlit app
├── requirements.txt      # Python dependencies
├── movies.csv            # Movie dataset (9,742 movies)
├── ratings.csv           # Ratings dataset (100,836 ratings)
├── static/
│   ├── style.css         # Main stylesheet (dark theme)
│   └── logo.png          # FrameForge logo
└── templates/
    ├── base.html         # Base template with navigation
    ├── index.html        # Home page with featured movies
    ├── mood.html         # Mood-based recommendations
    ├── search.html       # Search results page
    ├── top_rated.html    # Top rated movies page
    ├── signup.html       # User registration
    └── profile.html      # User profile with personalized picks
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Steps

1. **Clone/Navigate to the project directory**
   ```bash
   cd "movie recommender"
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add TMDB API Key** (for movie posters)
   - Get a free API key from [themoviedb.org](https://www.themoviedb.org/signup)
   - The key is already configured in `flask_app.py`

---

## ⚡ Quick Start

### Run the Flask Web Application
```bash
python flask_app.py
```
Open your browser to: **http://localhost:5000**

### Run the Core Recommender (CLI)
```bash
python main.py
```
This runs all recommendation algorithms and shows sample results.

### Run Evaluation Metrics
```bash
python evaluate.py
```
Outputs comprehensive metrics for all algorithms.

---

## 🌐 Web Application

### Pages

| Page | URL | Description |
|------|-----|-------------|
| **Home** | `/` | Featured movies and quick navigation |
| **Moods** | `/mood` | Browse movies by mood (19 moods available) |
| **Top Rated** | `/top-rated` | Highest-rated movies with filter options |
| **Search** | `/search?q=query` | Search movies by title |
| **Sign Up** | `/signup` | Create profile with genre preferences |
| **Profile** | `/profile` | Personalized recommendations |
| **Logout** | `/logout` | Clear session |

### Available Moods
- 😊 Happy, 😢 Sad, 🤩 Excited, 😌 Relaxed, 😱 Scary
- 💕 Romantic, 😂 Funny, 🤔 Thoughtful, 👨‍👩‍👧‍👦 Family
- 🕰️ Nostalgic, 🗺️ Adventurous, 🔍 Mysterious, ✨ Inspiring
- 🧊 Chill, 🌑 Dark, 💖 Heartwarming, ⚔️ Epic
- 🌀 Mind-bending, 🌈 Feel-good

---

## 📡 API Endpoints

### REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/recommend/mood/<mood>` | GET | Get recommendations for a mood |
| `/api/recommend/genres?genre=Action&genre=Comedy` | GET | Get recommendations by genres |
| `/api/search?q=query` | GET | Search movies by title |

### Example Response
```json
[
  {
    "movieId": 318,
    "title": "Shawshank Redemption, The (1994)",
    "genres": "Crime|Drama",
    "avg_rating": 4.43,
    "num_ratings": 317,
    "poster_url": "https://image.tmdb.org/t/p/w342/q6y0Go1tsGEsmtFryDOJo3dEmqu.jpg"
  }
]
```

---

## 🧠 Recommendation Algorithms

### 1. Collaborative Filtering (User-Based)
```python
# Computes similarity between users using cosine similarity
# Recommends movies liked by similar users
recommend_movies_collaborative(user_id, n=5)
```

### 2. Matrix Factorization (SVD)
```python
# Uses TruncatedSVD to decompose user-movie matrix
# Finds latent factors for users and movies
# Components: 50 latent factors
```

### 3. Neural Collaborative Filtering (NCF)
```python
# Deep learning model with:
# - User embeddings (32-dim)
# - Movie embeddings (32-dim)
# - MLP layers: 128 → 64 → 32 → 16
# - Dropout: 0.2 for regularization
```

### 4. Content-Based (Genre Similarity)
```python
# Uses Jaccard similarity on genre sets
recommend_similar_movies(movie_id, n=5)
```

### 5. Mood-Based Mapping
```python
mood_genres = {
    'happy': ['Comedy', 'Animation'],
    'sad': ['Drama'],
    'excited': ['Action', 'Adventure'],
    'romantic': ['Romance'],
    'scary': ['Thriller', 'Horror'],
    # ... 14 more moods
}
```

---

## 📊 Dataset

Using the **MovieLens Latest Small** dataset:

| File | Contents |
|------|----------|
| `movies.csv` | 9,742 movies with titles and genres |
| `ratings.csv` | 100,836 ratings from 610 users |

### Movie Genres
Action, Adventure, Animation, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western

---

## 📈 Evaluation Metrics

The `evaluate.py` script calculates:

| Metric | Description |
|--------|-------------|
| **RMSE** | Root Mean Squared Error (rating prediction accuracy) |
| **MAE** | Mean Absolute Error |
| **Precision@K** | Fraction of recommended items that are relevant |
| **Recall@K** | Fraction of relevant items that are recommended |
| **NDCG@K** | Ranking quality (rewards relevant items appearing first) |
| **Coverage** | Fraction of catalog ever recommended |
| **Diversity** | Genre variety in recommendations |

Run evaluation:
```bash
python evaluate.py
```

---

## ⚙️ Configuration

### TMDB API (Movie Posters)
Located in `flask_app.py`:
```python
TMDB_API_KEY = 'your-api-key-here'
TMDB_IMAGE_BASE = 'https://image.tmdb.org/t/p/w342'
```

### Flask Settings
```python
app.run(debug=True, port=5000)
```

### NCF Model Hyperparameters
Located in `ncf_model.py`:
```python
embedding_dim = 32
mlp_layers = [128, 64, 32, 16]
dropout_rate = 0.2
epochs = 15
batch_size = 256
```

---

## 🛠️ Tech Stack

- **Backend**: Flask 3.0+
- **Frontend**: HTML5, CSS3, Jinja2 templates
- **ML/DL**: scikit-learn, TensorFlow/Keras
- **Data**: pandas, numpy
- **External API**: TMDB for movie posters

---

## 📝 License

This project is for educational purposes. The MovieLens dataset is provided by GroupLens Research.

---

## 👤 Author

Built with ❤️ using Python, Flask, and Machine Learning.
