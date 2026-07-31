import pandas as pd
import numpy as np
import joblib
import os
import re

# ==========================
# 1. Load Dataset
# ==========================
# keep_default_na=False ensures empty values remain empty strings instead of NaN
df = pd.read_csv("cleaned_movies_data.csv", keep_default_na=False)

print(f"Initial Dataset Shape: {df.shape}")

# ==========================
# 2. Select Required Columns
# ==========================
columns = [
    "title",
    "genres",
    "overview",
    "director",
    "Actors",
    "original_language",
    "release_year",
    "vote_average",
    "vote_count",
    "popularity",
    "runtime",
    "tmdbId"
]

# Retain existing columns from the selection list
df = df[[col for col in columns if col in df.columns]]

# ==========================
# 3. Clean Text & Build Tags
# ==========================
text_columns = ["genres", "overview", "director", "Actors"]

for col in text_columns:
    df[col] = df[col].astype(str)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Generate temporary clean columns for tag concatenation
for col in text_columns:
    df[f"clean_{col}"] = df[col].apply(clean_text)

# Combine metadata into a single tags column for TF-IDF feature extraction
df["tags"] = (
    df["clean_genres"] + " " +
    df["clean_overview"] + " " +
    df["clean_director"] + " " +
    df["clean_Actors"]
)

# Remove temporary clean columns
df = df.drop(columns=[f"clean_{col}" for col in text_columns])

# ==========================
# 4. Remove Duplicate Movies
# ==========================
df = df.drop_duplicates(subset="title").reset_index(drop=True)

print(f"Shape after removing title duplicates: {df.shape}")

# ==========================
# 5. Save Output Files
# ==========================
os.makedirs("models", exist_ok=True)

# Save clean dataset copy
df.to_csv("movies_clean.csv", index=False)

# Dump pickled DataFrame for recommender.py
joblib.dump(df, "models/movies.pkl", compress=3)

print("\nSuccessfully created 'models/movies.pkl' and 'movies_clean.csv'!")