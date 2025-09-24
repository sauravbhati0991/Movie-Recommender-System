import streamlit as st
import joblib
import pandas as pd
import requests
import re
import string
import os
import base64
import time

# Set page configuration
st.set_page_config(page_title="Movie Recommendation System", layout="wide")

# API configuration
api_key = "ef4c4264f94150dc54dd29043c58a126"

st.title("🎬 Movie Recommendation and Review Analysis System")

# Load data and models
try:
    movies_dict = joblib.load('Model/movies_data.joblib')
    similarity = joblib.load('Model/similarity.joblib')
    movies_title = pd.DataFrame(movies_dict)
    
    # Load sentiment analysis models if available
    sentiment_analysis_available = True
    if os.path.exists('Model/sentiment_analysis_model.pkl') and os.path.exists('Model/tfidf_vectorizer.pkl'):
        sentiment_model = joblib.load('Model/sentiment_analysis_model.pkl')
        vectorizer = joblib.load('Model/tfidf_vectorizer.pkl')
    else:
        sentiment_analysis_available = False
        st.warning("⚠️ Sentiment analysis model not found. Reviews will be displayed without sentiment analysis.")
    
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

def preprocess_text(text):
    """Preprocess text for sentiment analysis"""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def analyze_sentiment(review):
    """Analyze sentiment of a review"""
    if not sentiment_analysis_available:
        return "neutral"
    
    if not review or not isinstance(review, str):
        return "neutral"
    
    try:
        processed_review = preprocess_text(review)
        if not processed_review.strip():
            return "neutral"
            
        vectorized_review = vectorizer.transform([processed_review])
        sentiment = sentiment_model.predict(vectorized_review)[0]
        return "positive" if sentiment == 1 else "negative"
    except Exception:
        return "neutral"

def get_movie_reviews(movie_id, movie_title):
    """Get reviews with fallback to AI-generated reviews"""
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}/reviews?api_key={api_key}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            reviews = []
            
            for review in data.get("results", []):
                author = review.get("author", "Anonymous")
                content = review.get("content", "")
                if content.strip():
                    reviews.append({"author": author, "content": content, "source": "TMDB"})
            
            if reviews:
                return reviews
        
        # Fallback to AI-generated reviews
        return generate_ai_reviews(movie_title, movie_id)
        
    except Exception:
        return generate_ai_reviews(movie_title, movie_id)

def generate_ai_reviews(movie_title, movie_id):
    """Generate realistic AI-style reviews"""
    try:
        movie_data = movies_title[movies_title['id'] == movie_id].iloc[0]
        rating = movie_data['vote_average']
        genres = movie_data['genres']
        
        if rating >= 8.0:
            reviews = [
                {
                    "author": "Film Critic",
                    "content": f"'{movie_title}' is a cinematic masterpiece! Brilliant direction and outstanding performances make this an unforgettable experience.",
                    "source": "AI Generated"
                },
                {
                    "author": "Movie Enthusiast",
                    "content": f"Absolutely loved {movie_title}! The storytelling is exceptional and the cinematography is breathtaking.",
                    "source": "AI Generated"
                }
            ]
        elif rating >= 6.5:
            reviews = [
                {
                    "author": "Casual Viewer",
                    "content": f"Really enjoyed {movie_title}! Great entertainment with solid performances and an engaging plot.",
                    "source": "AI Generated"
                },
                {
                    "author": "Movie Buff", 
                    "content": f"{movie_title} delivers quality entertainment. Well worth watching for fans of the genre.",
                    "source": "AI Generated"
                }
            ]
        else:
            reviews = [
                {
                    "author": "Critical Viewer",
                    "content": f"{movie_title} shows potential but has some uneven moments. Still offers decent entertainment.",
                    "source": "AI Generated"
                },
                {
                    "author": "Selective Watcher",
                    "content": f"Mixed feelings about {movie_title}. Some good elements but overall could be better.",
                    "source": "AI Generated"
                }
            ]
        
        return reviews
    except Exception:
        return [
            {
                "author": "Movie Fan",
                "content": f"Interesting take on the subject matter. {movie_title} offers a unique perspective worth exploring.",
                "source": "AI Generated"
            }
        ]

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
            st.write(f"**🎬 Director:** {movie_data['crew']}")
            
            cast_members = movie_data['cast'].split(", ")[:4]
            st.write(f"**👥 Cast:** {', '.join(cast_members)}")
        
        # Movie overview
        st.subheader("📖 Overview")
        st.info(movie_data['overview'])
        
        # Reviews section
        st.subheader("💬 Reviews")
        with st.spinner("Loading reviews..."):
            reviews = get_movie_reviews(movie_id, selected_movie)
        
        if reviews:
            for i, review in enumerate(reviews):
                with st.expander(f"Review by {review['author']} ({review.get('source', 'TMDB')})", expanded=(i == 0)):
                    st.write(review['content'])
                    if sentiment_analysis_available:
                        sentiment = analyze_sentiment(review['content'])
                        if sentiment == "positive":
                            st.success(f"**Sentiment:** {sentiment.title()} 👍")
                        elif sentiment == "negative":
                            st.error(f"**Sentiment:** {sentiment.title()} 👎")
                        else:
                            st.info(f"**Sentiment:** {sentiment.title()} 😐")
        
        # Recommendations section
        st.subheader("🎯 Similar Movies You Might Like")
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
    <p>🎬 <strong>Movie Recommendation and Review Analysis System</strong></p>
    <p>Built with Streamlit • TMDB API • Sentiment Analysis</p>
</div>
""", unsafe_allow_html=True)