import streamlit as st
import joblib
import pandas as pd
import requests
import re
import string
import os
from scipy.sparse import load_npz
import numpy as np

api_key = os.environ.get('API_KEY')

st.title("Movie Recommendation and Review Analysis System") 

# Load optimized files instead of original large files
try:
    # Load the minimal movies data
    movies_data = joblib.load('optimized_model_movies.joblib')
    
    # Load the sparse similarity matrix
    similarity_sparse = load_npz('optimized_model_similarity.npz')
    
    # Convert movies data back to DataFrame for compatibility
    movies_title = pd.DataFrame(movies_data)
    
except FileNotFoundError:
    st.error("Optimized model files not found. Please generate them first.")
    st.stop()

sentiment_model = joblib.load('Model/sentiment_analysis_model.pkl')
vectorizer  = joblib.load('Model/tfidf_vectorizer.pkl')

def search_movie_title(movie_data):
    select = st.selectbox("Search", movie_data["title"].values)
    return select

select = search_movie_title(movies_title)

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove links (http/https)
    text = re.sub(r'http\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Function to perform sentiment analysis
def analyze_sentiment(review):
    # Preprocess the review text and transform it using the vectorizer
    processed_review = preprocess_text(review)
    vectorized_review = vectorizer.transform([processed_review])
    
    # Predict sentiment using the loaded model
    sentiment = sentiment_model.predict(vectorized_review)[0]

    # Convert the sentiment prediction to a label (positive/negative)
    if sentiment == 1:
        sentiment_label = "positive"
    else:
        sentiment_label = "negative"

    return sentiment_label

def poster(movie_id):
    get_img = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-Us")
    data = get_img.json()
    
    if 'poster_path' in data:
        return "https://image.tmdb.org/t/p/original/" + data['poster_path']
    else:
        return "No poster available"
    
def get_movie_reviews(movie_id):
    reviews_url = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}/reviews?api_key={api_key}") 
    data = reviews_url.json()
    reviews = []
    
    for review in data.get("results", []):
        author = review.get("author", "Unknown")
        content = review.get("content", "")
        reviews.append({"author": author, "content": content})
    
    return reviews

# UPDATED RECOMMEND FUNCTION for optimized files
def recommend(movie):
    if movie not in movies_data['titles']:
        return [], []
    
    movie_index = movies_data['titles'].index(movie)
    
    # Convert sparse row to dense for that specific movie
    distances = similarity_sparse[movie_index].toarray().flatten()
    
    # Get top 10 recommendations (excluding the movie itself)
    top_indices = np.argsort(distances)[-11:-1][::-1]  # Get top 10 excluding self
    
    recommended_movies = []
    recommended_movies_posters = []
    
    for idx in top_indices:
        movie_id = movies_data['ids'][idx]
        recommended_movies.append(movies_data['titles'][idx])
        recommended_movies_posters.append(poster(movie_id))
    
    return recommended_movies, recommended_movies_posters

if select:
    names, posters = recommend(select)
    
    # Get movie details from the optimized data
    if select in movies_data['titles']:
        movie_index = movies_data['titles'].index(select)
        id = movies_data['ids'][movie_index]
        overview = movies_data['overview'][movie_index] if 'overview' in movies_data else "No overview available"
        genre = movies_data['genres'][movie_index]
        casts = movies_data['cast'][movie_index] if 'cast' in movies_data else "Cast information not available"
        crew = movies_data['crew'][movie_index] if 'crew' in movies_data else "Director information not available"
        release_date = movies_data['release_date'][movie_index] if 'release_date' in movies_data else "Release date not available"
        rating = movies_data['vote_average'][movie_index]
    else:
        st.error("Movie details not found")
        st.stop()
    
    # Create two columns layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(poster(id), use_column_width=True, width='100%')
    
    with col2:
        st.write(f'### {select}')
        st.write(f"**Genre:** {genre}")
        st.write(f"**Cast:** {casts}")
        st.write(f"**Director:** {crew}")
        st.write(f"**Release Date:** {release_date}")
        st.write(f"**Rating:** `{rating}`")
        
    st.warning(overview)

    # Fetch and display movie reviews
    st.write("#### Reviews:")
    movie_reviews = get_movie_reviews(id)
    
    # Create a scrollable container for reviews
    with st.container():
        
        if not movie_reviews:
            st.write("No reviews available.")
        else:
            # Display the first review
            st.write(f"**Author:** {movie_reviews[0]['author']}")
            st.write(movie_reviews[0]['content'])
            sentiment = analyze_sentiment(movie_reviews[0]['content'])
            
            # Display sentiment with emoji
            if sentiment == 'positive':
                emoji_html = '<span style="font-size: 25px;">👍</span>'
                st.write(f"##### `Review:` {sentiment.capitalize()} {emoji_html}", unsafe_allow_html=True)
            else:
                emoji_html = '<span style="font-size: 25px;">👎</span>'
                st.write(f"##### `Review:` {sentiment.capitalize()} {emoji_html}", unsafe_allow_html=True)

            st.write("-" * 40)

        # Flag to track if all reviews have been shown
        all_reviews_shown = False

        # Load more reviews on button click
        if len(movie_reviews) > 1:
            if not all_reviews_shown:
                if st.button("Show More Reviews"):
                    for review in movie_reviews[1:]:
                        st.write(f"**Author:** {review['author']}")
                        st.write(review['content'])
                        sentiment = analyze_sentiment(review['content'])
                        # Display sentiment with emoji
                        if sentiment == 'positive':
                            emoji_html = '<span style="font-size: 25px;">👍</span>'
                            st.write(f"##### `Review:` {sentiment.capitalize()} {emoji_html}", unsafe_allow_html=True)
                        else:
                            emoji_html = '<span style="font-size: 25px;">👎</span>'
                            st.write(f"##### `Review:` {sentiment.capitalize()} {emoji_html}", unsafe_allow_htm=True)
                        st.write("-" * 40)
                    
                    all_reviews_shown = True
        
        # Hide the button if all reviews have been shown
        if all_reviews_shown:
            st.button("Show Less Reviews", key="hide_button")

    st.write("#### Recommendations:")
    
    num_cols = 5  # Adjust the number of columns as needed
    cols = st.columns(num_cols)
    
    image_size = (135, 220)  # Adjusted image size
    
    # Add JavaScript for movie selection
    st.markdown("""
    <script>
    function selectMovie(movieName) {
        // This function will be called when a movie poster is clicked
        // You can implement navigation or selection logic here
        console.log("Selected movie: " + movieName);
        // For now, just log to console - you can add actual navigation logic
    }
    </script>
    """, unsafe_allow_html=True)

    for i in range(len(names)):
        with cols[i % num_cols]:
            # Create a clickable button overlaying the image using HTML/CSS
            button_html = f"""
                <div style="position: relative; padding-button: 20px; width: {image_size[0]}px; height: {image_size[1]}px;">
                <img  src="{posters[i]}" alt="{names[i]}" style="width: 100%; height: 100%; object-fit: cover;">{names[i]}
                    <button style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: transparent; border: none; cursor: pointer;" onclick="selectMovie('{names[i]}')"></button>
                </div>
            """
            st.markdown(button_html, unsafe_allow_html=True)
            st.markdown(" <br><br><br><br> ", unsafe_allow_html=True)

# Footer
st.markdown('''
--------------------------------------------------------------
🎬 Movie Recommendation & Sentiment Analysis 🍿

Explore, Analyze, and Discover Movies with Streamlit!

🌟Machine Learning and Natural Language Processing 🤖

Created by [Mukesh Mushyakhwo]

Contact: [mukesh@mukeshmushyakhwo.com.np]

GitHub: [https://github.com/MukeshMushyakhwo]

📧 Feel free to reach out for questions, feedback, or collaboration opportunities.

Happy Movie Watching! 🎥🍿
--------------------------------------------------------------
''')