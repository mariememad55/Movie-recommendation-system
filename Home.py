import streamlit as st

# MUST BE AT THE VERY TOP OF THE FILE
st.set_page_config(
    page_title="Movie AI System",
    page_icon="🎬",
    layout="wide"
)

import pandas as pd
import numpy as np
import joblib

# أي imports خاصة بالـ Recommendation
from recommender import *
from tmdb import *

# ===========================
# Navigation
# ===========================
if "page" not in st.session_state:
    st.session_state.page = "home"

# ===========================
# Home Page
# ===========================
if st.session_state.page == "home":

    st.markdown("""
<style>

.stApp{
    background:#0E1117;
    color:white;
}

.main-title{
    text-align:center;
    font-size:48px;
    font-weight:800;
    color:white;
    margin-top:30px;
}

/* H1 */
h1{
    color:#9ab !important;
    font-weight:800;
}

/* H2 */
h2{
    color:#9ab !important;
}

/* H3 */
h3{
    color:#9ab !important;
}


.sub-title{
    text-align:center;
    color:#BBBBBB;
    font-size:20px;
    margin-bottom:40px;
}

.home-card{
    background:#1B1F2A;
    padding:30px;
    border-radius:18px;
    box-shadow:0 0 20px rgba(255,75,75,.15);
}


 /* Hide Streamlit header */
            header {
                visibility: hidden;
            }

            /* Hide Deploy button and toolbar */
            [data-testid="stHeader"] {
                display: none;
            }

            /* Hide Main Menu */
            #MainMenu {
                visibility: hidden;
            }

            /* Hide Footer */
            footer {
                visibility: hidden;
            }

            /* Remove top spacing */
            .block-container {
                padding-top: 1rem;
            }

 .stButton > button {
                    background-color: #E53935 !important;
                    color: white !important;
                    border: none !important;
                    border-radius: 10px !important;
                    font-weight: bold !important;
                    transition: 0.3s;
                }

                /* عند مرور الماوس */
                .stButton > button:hover {
                    background-color: #C62828 !important;
                    color: white !important;
                    border: none !important;
                    transform:translateY(-2px);
                 
                }

                /* أثناء الضغط */
                .stButton > button:active {
                    background-color: #B71C1C !important;
                    border: none !important;
                }

                /* بعد الضغط (إلغاء اللون الأخضر) */
                .stButton > button:focus,
                .stButton > button:focus-visible {
                    outline: none !important;
                    box-shadow: 0 0 10px rgba(229,57,53,0.5) !important;
                    border: none !important;
                }

</style>
""", unsafe_allow_html=True)

    st.title("🎬 Movie AI System")

    st.markdown("## Welcome")

    st.write(
        "Choose the system you want to use."
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🎬 Recommendation System", use_container_width=True):
            st.session_state.page = "recommendation"
            st.rerun()

    with col2:
        if st.button("💰 Revenue Prediction", use_container_width=True):
            st.session_state.page = "revenue"
            st.rerun()

# ===========================
# Recommendation
# ===========================
elif st.session_state.page == "recommendation":
            if st.button("⬅ Back to Home"):
               st.session_state.page = "home"
               st.rerun()

            st.divider()

            # Load external style.css if present
            try:
                with open("style.css", "r") as f:
                    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
            except FileNotFoundError:
                pass

            st.markdown("""
            <style>
                .stApp {
                    background-color: #14181c;
                    color: #9ab;
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                }

                   h1{
                    color:#9ab !important;
                     font-weight:800;
                    }

                    
                    h2{
                        color:#9ab !important;
                    }

                   
                    h3{
                        color:#9ab !important;
                    }

                
                
                    .stButton > button {
                    background-color: #E53935 !important;
                    color: white !important;
                    border: none !important;
                    border-radius: 10px !important;
                    font-weight: bold !important;
                    transition: 0.3s;
                }

                
                .stButton > button:hover {
                    background-color: #C62828 !important;
                    color: white !important;
                }

                
                .stButton > button:active {
                    background-color: #B71C1C !important;
                }

                
                 .stButton > button:focus,
                            .stButton > button:focus-visible {
                            outline: none !important;
                            box-shadow: 0 0 12px rgba(255, 75, 75, 0.7) !important;
                              }
                
                
                
                .movie-title {
                    color: #ffffff;
                    font-weight: 700;
                    font-size: 1rem;
                    margin-top: 6px;
                    margin-bottom: 2px;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                }
                
                .since-liked-tag {
                    color: #00e054;
                    font-size: 0.85rem;
                    font-weight: 600;
                    margin-bottom: 6px;
                }
                
                .meta-text {
                    color: #89a;
                    font-size: 0.8rem;
                    margin-bottom: 2px;
                }

                div[data-baseweb="select"] {
                    background-color: #1e252c !important;
                    border-color: #2c3440 !important;
                }
                
                .stExpander {
                    background-color: #1a2129 !important;
                    border: 1px solid #2c3440 !important;
                    border-radius: 4px !important;
                    margin-top: 6px;
                }
            </style>
            """, unsafe_allow_html=True)


            # ---------------------------------------------------------
            # Helpers
            # ---------------------------------------------------------
            def safe_get_poster(tmdb_id):
                if pd.isna(tmdb_id) or str(tmdb_id).strip() == "" or tmdb_id is None:
                    return "https://via.placeholder.com/500x750?text=No+Poster"
                try:
                    clean_id = int(float(tmdb_id))
                    poster = get_poster(clean_id)
                    if poster:
                        return poster
                except Exception:
                    pass
                return "https://via.placeholder.com/500x750?text=No+Poster"


            def get_since_liked_reason(row, selected_movies, section_type=None):
                selected_rows = movies[movies["title"].isin(selected_movies)]
                
                r_dir = str(row.get("director", "")).strip().lower()
                r_actors = set([a.strip().lower() for a in str(row.get("Actors", "")).split(",") if a.strip()])
                r_genres = set([g.strip().lower() for g in str(row.get("genres", "")).split(",") if g.strip()])
                
                # 1. DIRECTOR SECTION FIX
                if section_type == "director":
                    for _, s_row in selected_rows.iterrows():
                        s_dir = str(s_row.get("director", "")).strip().lower()
                        if s_dir and s_dir == r_dir:
                            return f"Directed by {str(s_row.get('director', '')).title()} ({s_row['title']})"
                    return f"Directed by {str(row.get('director', '')).title()}"

                # 2. ACTOR SECTION FIX
                if section_type == "actor":
                    for _, s_row in selected_rows.iterrows():
                        s_actors = set([a.strip().lower() for a in str(s_row.get("Actors", "")).split(",") if a.strip()])
                        common_actors = r_actors.intersection(s_actors)
                        if common_actors:
                            actor_name = list(common_actors)[0].title()
                            return f"Stars {actor_name} ({s_row['title']})"

                # 3. GENRE / FALLBACK FIX
                for _, s_row in selected_rows.iterrows():
                    s_dir = str(s_row.get("director", "")).strip().lower()
                    s_actors = set([a.strip().lower() for a in str(s_row.get("Actors", "")).split(",") if a.strip()])
                    s_genres = set([g.strip().lower() for g in str(s_row.get("genres", "")).split(",") if g.strip()])
                    
                    if (s_dir and s_dir == r_dir) or s_actors.intersection(r_actors) or s_genres.intersection(r_genres):
                        return f"Based on {s_row['title']} in your favorites"

                return "Based on your favorite choices"


            def render_movie_card(movie_row, selected_movies=None, is_rec=False, section_type=None):
                poster_url = safe_get_poster(movie_row.get("tmdbId"))
                
                director = str(movie_row.get("director", "")).title() or "Unknown"
                cast = str(movie_row.get("Actors", "")).title() or "N/A"
                genres = str(movie_row.get("genres", "")).title() or "N/A"
                rating = movie_row.get("vote_average", movie_row.get("rating", "N/A"))
                
                year = movie_row.get("release_year", movie_row.get("year", "N/A"))
                if str(year).endswith(".0"):
                    year = str(year)[:-2]

                overview = str(movie_row.get("overview", "")).strip() or "No overview available."

                with st.container(border=True):
                    if is_rec and selected_movies:
                        since_liked = get_since_liked_reason(movie_row, selected_movies, section_type=section_type)
                        st.markdown(f"<div class='since-liked-tag'>{since_liked}</div>", unsafe_allow_html=True)
                        
                    st.image(poster_url, use_container_width=True)
                    st.markdown(f"<div class='movie-title'>{movie_row['title']}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='meta-text'>📅 <b>Year:</b> {year}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='meta-text'>🎬 <b>Director:</b> {director}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='meta-text'>🎭 <b>Cast:</b> {cast}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='meta-text'>🍿 <b>Genre:</b> {genres}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='meta-text'>⭐ <b>Rating:</b> {rating}/10</div>", unsafe_allow_html=True)
                    
                    with st.expander("📖 Overview"):
                        st.write(overview)


            def display_movie_grid(df_recs, selected_movies, section_type=None):
                if df_recs is None or df_recs.empty:
                    st.info("No recommendations found for this specific selection.")
                    return

                cols = st.columns(4)
                for i, (_, row) in enumerate(df_recs.iterrows()):
                    col = cols[i % 4]
                    with col:
                        render_movie_card(row, selected_movies=selected_movies, is_rec=True, section_type=section_type)


            # ---------------------------------------------------------
            # App Interface
            # ---------------------------------------------------------
            st.title("MOVIES RECOMMENDER")
            st.caption("Powered by TF-IDF Embeddings & TMDB API")

            st.markdown("---")

            def reset_recs():
                for key in ["recs_director", "recs_actor", "recs_genre"]:
                    if key in st.session_state:
                        del st.session_state[key]

            movie_list = sorted(movies["title"].tolist())
            selected_movies = st.multiselect(
                "Choose your top 3 favorite movies:",
                options=movie_list,
                default=movie_list[3:6] if len(movie_list) >= 3 else movie_list,
                max_selections=3,
                on_change=reset_recs
            )

            if len(selected_movies) == 3:
                m1, m2, m3 = selected_movies[0], selected_movies[1], selected_movies[2]
                
                st.subheader("Your Favorite Selection")
                fav_cols = st.columns(3)
                
                for col, m_title in zip(fav_cols, selected_movies):
                    match = movies[movies["title"] == m_title].iloc[0]
                    with col:
                        render_movie_card(match, is_rec=False)

                st.markdown("---")
                
                if st.button("RECOMMEND MOVIES"):
                    st.session_state["recs_director"] = recommend_by_director(m1, m2, m3, top_n=8)
                    st.session_state["recs_actor"] = recommend_by_actor(m1, m2, m3, top_n=8)
                    st.session_state["recs_genre"] = recommend_by_genre(m1, m2, m3, top_n=8)

                if "recs_director" in st.session_state:
                    st.subheader("🎬 Recommending Movies by Directors")
                    display_movie_grid(st.session_state["recs_director"], selected_movies, section_type="director")
                    
                    st.markdown("---")
                    
                    st.subheader("🎭 Recommending Movies by Cast / Actors")
                    display_movie_grid(st.session_state["recs_actor"], selected_movies, section_type="actor")
                    
                    st.markdown("---")
                    
                    st.subheader("🍿 Recommending Movies by Genre")
                    display_movie_grid(st.session_state["recs_genre"], selected_movies, section_type="genre")

            elif len(selected_movies) > 0:
                st.info("Please select exactly 3 movies to generate recommendations.")
            else:
                st.warning("Select 3 favorite movies from the dropdown above to begin.")


elif st.session_state.page == "revenue":

            if st.button("⬅ Back to Home"):
                st.session_state.page = "home"
                st.rerun()

            st.divider()

            st.markdown("""
            <style>

            /* جميع أزرار Streamlit */
            .stButton > button {
                background-color: #E53935 !important;
                color: white !important;
                border: none !important;
                border-radius: 10px !important;
                font-weight: bold !important;
                transition: 0.3s;
            }

            /* عند مرور الماوس */
            .stButton > button:hover {
               background-color: #C62828 !important;
               color: white !important;
               border: none !important;
                outline: none !important;
            
                 transform:translateY(-2px);
                                    
                 box-shadow:0px 6px 20px rgba(192, 16, 0, 0.35);
            }

            /* أثناء الضغط */
            .stButton > button:active {
                background-color: #B71C1C !important;
            }

            .stButton > button:focus,
            .stButton > button:focus-visible {
            outline: none !important;
            box-shadow: 0 0 12px rgba(255, 75, 75, 0.7) !important;
              }
            /* Hide Streamlit header */
            header {
                visibility: hidden;
            }

            /* Hide Deploy button and toolbar */
            [data-testid="stHeader"] {
                display: none;
            }

            /* Hide Main Menu */
            #MainMenu {
                visibility: hidden;
            }

            /* Hide Footer */
            footer {
                visibility: hidden;
            }

            /* Remove top spacing */
            .block-container {
                padding-top: 1rem;
            }

            
            /* Labels */
            label, .stSelectbox label, .stNumberInput label, .stSlider label{
                color:#9ab !important;
                font-size:17px !important;
                font-weight:600 !important;
            }
            .stApp {
                background-color: #14181c;
                color: #9ab;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                            }

            /* Number Input */
            div[data-baseweb="input"] input{
                background:#262730 !important;
                color:white !important;
                border-radius:10px !important;
                border:1px solid #444 !important;
            }

            /* Select Box */
            div[data-baseweb="select"]{
                background:#262730 !important;
                border-radius:10px !important;
            }

            div[data-baseweb="select"] > div{
                background:#262730 !important;
                color:white !important;
                border:1px solid #444 !important;
            }

            /* Dropdown menu */
            ul{
                background:#262730 !important;
                color:white !important;
            }

            /* Slider */
            .stSlider{
                padding-top:10px;
            }

            .stSlider label{
                color:white !important;
            }

            /* Sidebar */
            section[data-testid="stSidebar"]{
                background:#171923;
            }

            </style>
            """, unsafe_allow_html=True)

            st.title("🎬 Movie Revenue Prediction")

            st.write("Predict movie revenue using CatBoost")

            # ==============================
            # Load Model (UPDATED RELATIVE PATH)
            # ==============================

            model = joblib.load("models/catboost_movie_model.pkl")

            # ==============================
            # Load Dataset
            # ==============================

            df = pd.read_csv("cleaned_movies_data.csv")

            # ==============================
            # Extract Actors
            # ==============================

            actors = []

            for item in df["Actors"].dropna():

                actors.extend(
                    [x.strip() for x in str(item).split(",")]
                )

            actors = sorted(list(set(actors)))

            # ==============================
            # Directors
            # ==============================

            directors = sorted(
                df["director"].dropna().unique()
            )

            # ==============================
            # Genres
            # ==============================

            genres = sorted(
                df["genres"].dropna().unique()
            )

            # ==============================
            # Companies
            # ==============================

            companies = []

            for item in df["production_companies"].dropna():

                companies.extend(
                    [x.strip() for x in str(item).split(",")]
                )

            companies = sorted(list(set(companies)))

            # ==============================
            # Collections
            # ==============================

            collections = sorted(
                df["belongs_to_collection"].fillna("No Collection").unique()
            )

            # ==============================
            # Countries
            # ==============================

            countries = sorted(
                df["production_countries"].dropna().unique()
            )

            # ==============================
            # Languages
            # ==============================

            languages = sorted(
                df["original_language"].dropna().unique()
            )

            # ==============================
            # Sidebar
            # ==============================

            st.header("Movie Information")

            budget = st.sidebar.number_input(
                "Budget",
                min_value=0,
                value=50000000,
                step=1000000
            )

            popularity = st.sidebar.number_input(
                "Popularity",
                min_value=0.0,
                value=15.0
            )

            runtime = st.sidebar.number_input(
                "Runtime",
                min_value=30,
                max_value=300,
                value=120
            )

            release_year = st.sidebar.number_input(
                "Release Year",
                min_value=1980,
                max_value=2035,
                value=2026
            )

            release_month = st.sidebar.selectbox(
                "Release Month",
                list(range(1,13))
            )

            adult = st.sidebar.selectbox(
                "Adult",
                ["False","True"]
            )

            status = st.sidebar.selectbox(
                "Status",
                [
                    "Released",
                    "Post Production",
                    "In Production",
                    "Planned",
                    "Rumored"
                ]
            )

            vote_average = st.sidebar.slider(
                "Expected Vote Average",
                0.0,
                10.0,
                7.0
            )

            vote_count = st.sidebar.number_input(
                "Expected Vote Count",
                min_value=0,
                value=1000
            )
            # ==============================
            # Searchable Inputs
            # ==============================

            col1, col2 = st.columns(2)

            with col1:

                collection = st.selectbox(
                    "Movie Collection",
                    collections,
                    index=0
                )

                genre = st.selectbox(
                    "Genre",
                    genres
                )

                language = st.selectbox(
                    "Original Language",
                    languages
                )

                country = st.selectbox(
                    "Production Country",
                    countries
                )

                director = st.selectbox(
                    "Director",
                    directors
                )

            with col2:

                company1 = st.selectbox(
                    "Production Company 1",
                    companies,
                    key="company1"
                )

                company2 = st.selectbox(
                    "Production Company 2",
                    companies,
                    key="company2"
                )

                actor1 = st.selectbox(
                    "Lead Actor",
                    actors,
                    key="actor1"
                )

                actor2 = st.selectbox(
                    "Actor 2",
                    actors,
                    key="actor2"
                )

                actor3 = st.selectbox(
                    "Actor 3",
                    actors,
                    key="actor3"
                )

            # ==============================
            # Merge Inputs
            # ==============================

            actors_input = ", ".join(
                [x for x in [actor1, actor2, actor3] if x]
            )

            companies_input = ", ".join(
                [x for x in [company1, company2] if x]
            )

            st.divider()

            st.subheader("Movie Information")

            preview1, preview2 = st.columns(2)

            with preview1:

                st.write("**Director:**", director)
                st.write("**Actors:**", actors_input)
                st.write("**Genre:**", genre)
                st.write("**Collection:**", collection)

            with preview2:

                st.write("**Production Companies:**", companies_input)
                st.write("**Language:**", language)
                st.write("**Country:**", country)
                st.write("**Release:**", f"{release_month}/{release_year}")

            st.divider()

            predict = st.button(
                "🎬 Predict Revenue",
                use_container_width=True
            )

            # ===========================================
            # Prediction
            # ===========================================

            if predict:

                # Create dataframe بنفس ترتيب التدريب
                input_data = pd.DataFrame({

                    "adult":[adult],

                    "belongs_to_collection":[collection],

                    "budget":[budget],

                    "genres":[genre],

                    "original_language":[language],

                    "popularity":[popularity],

                    "production_companies":[companies_input],

                    "production_countries":[country],

                    "runtime":[runtime],

                    "status":[status],

                    "vote_average":[vote_average],

                    "vote_count":[vote_count],

                    "director":[director],

                    "Actors":[actors_input],

                    "release_year":[release_year],

                    "release_month":[release_month]

                })

                # Prediction
                prediction = model.predict(input_data)

                revenue = np.expm1(prediction[0])

                profit = revenue - budget

                st.success("Prediction Completed Successfully!")

                c1, c2 = st.columns(2)

                with c1:

                    st.metric(
                        label="💰 Predicted Revenue",
                        value=f"${revenue:,.0f}"
                    )

                with c2:

                    st.metric(
                        label="📈 Estimated Profit",
                        value=f"${profit:,.0f}"
                    )

                st.progress(100)

                st.divider()

                st.subheader("Movie Summary")

                summary = pd.DataFrame({

                    "Feature":[
                        "Budget",
                        "Genre",
                        "Collection",
                        "Director",
                        "Actors",
                        "Production Companies",
                        "Language",
                        "Country",
                        "Runtime",
                        "Popularity",
                        "Release Year",
                        "Release Month"
                    ],

                    "Value":[
                        budget,
                        genre,
                        collection,
                        director,
                        actors_input,
                        companies_input,
                        language,
                        country,
                        runtime,
                        popularity,
                        release_year,
                        release_month
                    ]

                })

                st.dataframe(
                    summary,
                    use_container_width=True,
                    hide_index=True
                )

                st.balloons()