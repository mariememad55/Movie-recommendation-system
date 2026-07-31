import requests

# ===========================================
# TMDb API Key
# ===========================================
API_KEY = "e9f9423c95bca1f49793bdfb542cfd75"

BASE_URL = "https://api.themoviedb.org/3/movie/"
IMAGE_URL = "https://image.tmdb.org/t/p/w500"

# ===========================================
# Get Movie Details
# ===========================================
def get_movie_details(tmdb_id):
    if tmdb_id is None or str(tmdb_id).strip() == "":
        return None

    try:
        url = f"{BASE_URL}{int(float(tmdb_id))}"
        params = {
            "api_key": API_KEY,
            "language": "en-US"
        }
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return None
        return response.json()
    except Exception:
        return None


# ===========================================
# Get Poster URL
# ===========================================
def get_poster(tmdb_id):
    movie = get_movie_details(tmdb_id)
    if movie is None:
        return None

    poster_path = movie.get("poster_path")
    if poster_path is None:
        return None

    return IMAGE_URL + poster_path


# ===========================================
# Get Movie Information
# ===========================================
def get_movie_info(tmdb_id):
    movie = get_movie_details(tmdb_id)
    if movie is None:
        return None

    return {
        "title": movie.get("title"),
        "poster": IMAGE_URL + movie["poster_path"] if movie.get("poster_path") else None,
        "rating": movie.get("vote_average"),
        "release_date": movie.get("release_date"),
        "runtime": movie.get("runtime"),
        "genres": ", ".join([g["name"] for g in movie.get("genres", [])]),
        "overview": movie.get("overview")
    }