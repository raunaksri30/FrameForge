# FrameForge - Movie Recommender System

FrameForge is a comprehensive movie recommendation platform designed to transform how users discover and watch films. Built with Flask and Python, it leverages machine learning algorithms to provide personalized suggestions based on over 100,000 user ratings spanning from 1996 to 2018 and genre preferences. The application features a dynamic, dark-themed interface where users can browse top-rated content, explore movies by mood, and search a vast database of over 9,700 titles. With secure user authentication, real-time poster integration via the TMDB API, and direct movie streaming capabilities, FrameForge offers a seamless, end-to-end entertainment experience from discovery to playback.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)
![SQLite](https://img.shields.io/badge/SQLite-Database-lightblue.svg)

---

## Demo

> *Demo video will be added here*

---

## Features

### Core Functionality
- **Movie Streaming** - Watch movies directly with integrated video player
- **Mood-Based Discovery** - Find movies matching your current mood (19 moods available)
- **Smart Search** - Search through 9,700+ movies by title
- **User Authentication** - Secure signup/login with password hashing
- **Personalized Recommendations** - Get suggestions based on your genre preferences

### Technical Features
- Real-time movie posters from TMDB API
- SQLite database for persistent user storage
- Responsive dark-themed UI
- RESTful API endpoints

---

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables**
   
   Create a `.env` file in the project root:
   ```
   TMDB_API_KEY=your_tmdb_api_key_here
   ```
   Get a free API key from [themoviedb.org](https://www.themoviedb.org/signup)

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Open in browser**
   ```
   http://localhost:5000
   ```

---

## Project Structure

```
movie-recommender/
├── app.py                # Main Flask application
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (not tracked)
├── .gitignore            # Git ignore rules
├── users.db              # SQLite database (not tracked)
├── movies.csv            # Movie dataset (9,742 movies)
├── ratings.csv           # User ratings (100,836 ratings)
├── static/
│   └── style.css         # Stylesheet
└── templates/
    ├── base.html         # Base template
    ├── index.html        # Home page
    ├── mood.html         # Mood-based recommendations
    ├── search.html       # Search results
    ├── watch.html        # Movie streaming page
    ├── signup.html       # User registration
    ├── login.html        # User login
    └── profile.html      # User profile
```

---

## Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Home page with featured movies |
| `/mood` | GET | Mood-based movie discovery |
| `/top-rated` | GET | Highest-rated movies |
| `/search` | GET | Search movies by title |
| `/watch/<id>` | GET | Stream movie |
| `/signup` | GET/POST | User registration |
| `/login` | GET/POST | User login |
| `/profile` | GET | Personalized recommendations |
| `/logout` | GET | Log out user |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/recommend/mood/<mood>` | GET | Get mood-based recommendations |
| `/api/recommend/genres` | GET | Get genre-based recommendations |
| `/api/search` | GET | Search movies |

---

## Security

- **Password Hashing**: User passwords are hashed using Werkzeug
- **Environment Variables**: Sensitive data stored in `.env` file
- **Gitignored Files**: `.env` and `users.db` are not tracked in version control

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Flask 3.0, SQLAlchemy |
| Database | SQLite |
| Frontend | HTML5, CSS3, JavaScript |
| Authentication | Werkzeug Security |
| APIs | TMDB (posters), Vidking (streaming) |

---

## Deployment

### Environment Variables
Set the following on your hosting platform:
```
TMDB_API_KEY=your_api_key
```

### Production Server
```bash
gunicorn app:app
```

---

## Dataset

Using the MovieLens Latest Small dataset:
- 9,742 movies with titles and genres
- 100,836 ratings from 610 users

---

## License

For educational purposes. MovieLens dataset provided by GroupLens Research.
