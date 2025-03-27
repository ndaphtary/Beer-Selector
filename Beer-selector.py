import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st

# Load dataset
beer_data = pd.read_csv('beer_profile_and_ratings.csv')
beer_data = beer_data[['Beer Name (Full)', 'Style', 'review_overall']]

# Drop rows with missing styles
beer_data = beer_data.dropna(subset=['Style'])

# Compute TF-IDF vectors for styles (optional: could be used to extend recommendations)
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(beer_data['Style'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Streamlit App
st.set_page_config(page_title="🍺 Top Beers by Style", layout="wide")
st.title("🍺 Beer Recommender by Style")
st.markdown("Select a beer style to view top-rated beers based on user reviews.")

# Dropdown for style selection
styles = sorted(beer_data['Style'].dropna().unique())
selected_style = st.selectbox("Choose your beer style:", styles)

# Function to get top-rated beers
def get_top_rated_beers(style, num_recommendations=10):
    style_beer = beer_data[beer_data['Style'] == style]
    top_rated = style_beer.sort_values(by='review_overall', ascending=False).head(num_recommendations)
    return top_rated[['Beer Name (Full)', 'review_overall']]

# Show results
if selected_style:
    results = get_top_rated_beers(selected_style)
    st.subheader(f"Top Beers in Style: {selected_style}")
    st.dataframe(results.reset_index(drop=True))
