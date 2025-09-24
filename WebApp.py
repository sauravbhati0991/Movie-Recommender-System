import streamlit as st
import joblib
import pandas as pd
import requests
import os
import time
import base64

# Set page configuration
st.set_page_config(page_title="Movie Recommendation System", layout="wide")

# API configuration
api_key = "ef4c4264f94150dc54dd29043c58a126"

st.title("🎬 Movie Recommendation System")

# Load data and models
try:
    movies_dict = joblib.load('Model/movies_data.joblib')
    similarity = joblib.load('Model/similarity.joblib')
    movies_title = pd.DataFrame(movies_dict)
    st.success("✅ Models loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading files: {e}")
    st.stop()

def generate_blank_card(movie_title, width=300, height=450):
    """Generate a simple blank card with movie title"""
    svg = f'''
    <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
        <rect width="100%" height="100%" fill="#f0f2f6" stroke="#ddd" stroke-width="2"/>
        <rect x="10%" y="10%" width="80%" height="60%" fill="#e0e0e0" stroke="#ccc" stroke-width="1"/>
        <text x="50%" y="85%" font-family="Arial" font-size="14" fill="#333" 
              text-anchor="middle" font-weight="bold">{movie_title}</text>
    </svg>
    '''
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    return f"data:image/svg+xml;base64,{b64}"

def fetch_poster(movie_id, movie_title):
    """Try to fetch poster from TMDB, fallback to blank card"""
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            poster_path = data.get('poster_path')
            
            if poster_path:
                poster_url = f"https://image.tmdb.org/t/p/w500/{poster_path}"
                head_response = requests.head(poster_url, timeout=5)
                if head_response.status_code == 200:
                    return poster_url
        
        return generate_blank_card(movie_title)
        
    except Exception:
        return generate_blank_card(movie_title)

def recommend_movies(movie_title):
    """Get movie recommendations with similarity scores"""
    try:
        movie_match = movies_title[movies_title['title'] == movie_title]
        if len(movie_match) == 0:
            return [], [], []
        
        movie_index = movie_match.index[0]
        similarities = similarity[movie_index]
        similar_movies = sorted(list(enumerate(similarities)), reverse=True, key=lambda x: x[1])[1:11]
        
        recommendations = []
        posters = []
        similarity_scores = []
        
        for idx, score in similar_movies:
            movie_id = movies_title.iloc[idx].id
            movie_title_rec = movies_title.iloc[idx].title
            recommendations.append(movie_title_rec)
            similarity_scores.append(round(score * 100, 1))
            
            poster_url = fetch_poster(movie_id, movie_title_rec)
            posters.append(poster_url)
        
        return recommendations, posters, similarity_scores
    except Exception as e:
        st.error(f"Error getting recommendations: {e}")
        return [], [], []

# Sidebar for movie selection
st.sidebar.title("🔍 Movie Search")
selected_movie = st.sidebar.selectbox(
    "Choose a movie:", 
    movies_title["title"].values,
    index=0
)



# Main content
if selected_movie:
    try:
        movie_data = movies_title[movies_title['title'] == selected_movie].iloc[0]
        movie_id = movie_data['id']
        
        # Display movie details
        col1, col2 = st.columns([1, 2])
        
        with col1:
            poster_url = fetch_poster(movie_id, selected_movie)
            st.image(poster_url, use_container_width=True)
            
        with col2:
            st.header(selected_movie)
            rating = movie_data['vote_average']
            stars = "⭐" * min(5, int(rating / 2))
            st.write(f"**{stars} {rating}/10**")
            st.write(f"**🎭 Genres:** {movie_data['genres']}")
            st.write(f"**📅 Release Date:** {movie_data['release_date']}")
        
        # Movie overview
        st.subheader("📖 Overview")
        st.write(movie_data['overview'])
        
        # Recommendations section
        st.subheader("🎯 Similar Movies")
        with st.spinner("Finding similar movies..."):
            recommendations, posters, similarity_scores = recommend_movies(selected_movie)
        
        if recommendations:
            st.write(f"Found **{len(recommendations)}** similar movies:")
            cols = st.columns(5)
            
            for i, (movie, poster_url, similarity) in enumerate(zip(recommendations, posters, similarity_scores)):
                with cols[i % 5]:
                    st.image(poster_url, use_container_width=True)
                    st.write(f"**{movie}**")
                    st.write(f"**Similarity:** {similarity}%")
        else:
            st.info("No recommendations available.")
            
    except Exception as e:
        st.error(f"Error displaying movie details: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center;">
    <p>🎬 <strong>Movie Recommendation System</strong></p>
    <p>Built with Streamlit • TMDB API</p>
</div>
""", unsafe_allow_html=True)