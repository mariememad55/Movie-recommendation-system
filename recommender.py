import pandas as pd
import numpy as np
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# LOAD DATA & PREPARE MATRICES
# =====================================================
movies = joblib.load("movies.pkl")

movies["overview"] = movies["overview"].astype(str)
movies["genres"] = movies["genres"].astype(str)
movies["Actors"] = movies["Actors"].astype(str)
movies["director"] = movies["director"].astype(str)

overview_vectorizer = TfidfVectorizer(stop_words="english", max_features=8000)
genre_vectorizer = TfidfVectorizer()
actor_vectorizer = TfidfVectorizer()

overview_matrix = overview_vectorizer.fit_transform(movies["overview"])
genre_matrix = genre_vectorizer.fit_transform(movies["genres"])
actor_matrix = actor_vectorizer.fit_transform(movies["Actors"])


# =====================================================
# HELPER FUNCTIONS
# =====================================================
def get_indices(selected_movies):
    indices = []
    for movie in selected_movies:
        match = movies[movies["title"] == movie]
        if match.empty:
            continue
        indices.append(match.index[0])
    return indices


def build_dataframe(scores, selected_movies, top_n=8):
    sorted_idx = np.argsort(scores)[::-1]
    recommendations = []

    for idx in sorted_idx:
        row = movies.iloc[idx]

        # Exclude selected input movies and non-matching scores
        if row["title"] in selected_movies or scores[idx] <= 0:
            continue

        recommendations.append({
            "title": row["title"],
            "tmdbId": row["tmdbId"],
            "genres": row["genres"],
            "rating": row.get("vote_average", "N/A"),
            "year": row.get("release_year", "N/A"),
            "runtime": row.get("runtime", "N/A"),
            "director": row["director"],
            "Actors": row["Actors"],
            "overview": row["overview"],
            "score": float(scores[idx])
        })

        if len(recommendations) >= top_n:
            break

    return pd.DataFrame(recommendations)


# =====================================================
# RECOMMENDATION FUNCTIONS (TOP 8)
# =====================================================
def recommend_by_director(movie1, movie2, movie3, top_n=8):
    selected_movies = [movie1, movie2, movie3]
    indices = get_indices(selected_movies)

    if len(indices) < 3:
        return pd.DataFrame()

    selected_directors = set(
        str(movies.iloc[idx]["director"]).strip().lower() 
        for idx in indices 
        if str(movies.iloc[idx]["director"]).strip()
    )

    scores = np.array([
        1.0 if str(director).strip().lower() in selected_directors and str(director).strip() != "" else 0.0
        for director in movies["director"]
    ])

    return build_dataframe(scores, selected_movies, top_n=top_n)


def recommend_by_actor(movie1, movie2, movie3, top_n=8):
    selected_movies = [movie1, movie2, movie3]
    indices = get_indices(selected_movies)

    if len(indices) < 3:
        return pd.DataFrame()

    profile = np.asarray(actor_matrix[indices].mean(axis=0))
    similarity_scores = cosine_similarity(profile, actor_matrix).flatten()

    return build_dataframe(similarity_scores, selected_movies, top_n=top_n)


def recommend_by_genre(movie1, movie2, movie3, top_n=8):
    selected_movies = [movie1, movie2, movie3]
    indices = get_indices(selected_movies)

    if len(indices) < 3:
        return pd.DataFrame()

    profile = np.asarray(genre_matrix[indices].mean(axis=0))
    similarity_scores = cosine_similarity(profile, genre_matrix).flatten()

    return build_dataframe(similarity_scores, selected_movies, top_n=top_n)