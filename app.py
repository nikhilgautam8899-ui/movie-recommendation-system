import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader

st.title("🎬 Hybrid Movie Recommendation System")

@st.cache_data
def load_data():
    movies = pd.read_csv('ml-latest-small/movies.csv')
    ratings = pd.read_csv('ml-latest-small/ratings.csv')
    tags = pd.read_csv('ml-latest-small/tags.csv')

    tags_combined = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
    movies = movies.merge(tags_combined, on='movieId', how='left')
    movies['tag'] = movies['tag'].fillna('')
    movies['genres_clean'] = movies['genres'].str.replace('|', ' ', regex=False)
    movies['features'] = movies['genres_clean'] + ' ' + movies['tag']

    return movies, ratings

@st.cache_resource
def build_models(movies, ratings):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['features'])
    content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    svd_model = SVD()
    svd_model.fit(trainset)

    return content_sim, svd_model

movies, ratings = load_data()
content_sim, svd_model = build_models(movies, ratings)
indices = pd.Series(movies.index, index=movies['title'].str.lower())

def hybrid_recommend(title, user_id, top_n=10):
    title = title.lower()
    idx = indices[title]
    sim_scores = list(enumerate(content_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:50]
    movie_indices = [i[0] for i in sim_scores]
    candidates = movies.iloc[movie_indices].copy()
    candidates['content_score'] = [i[1] for i in sim_scores]
    candidates['predicted_rating'] = candidates['movieId'].apply(
        lambda x: svd_model.predict(user_id, x).est
    )
    candidates['hybrid_score'] = (candidates['content_score'] * 0.5) + \
                                   (candidates['predicted_rating'] / 5 * 0.5)
    candidates = candidates.sort_values('hybrid_score', ascending=False)
    return candidates[['title', 'predicted_rating']].head(top_n)

st.write("Movie choose karo aur apna User ID daalo - personalized recommendations milenge!")

movie_list = movies['title'].values
selected_movie = st.selectbox("Movie select karo:", movie_list)
user_id = st.number_input("User ID daalo:", min_value=1, max_value=int(ratings['userId'].max()), value=1)

if st.button("Recommend karo"):
    results = hybrid_recommend(selected_movie, user_id)
    st.subheader("Tumhare liye recommendations:")
    for i, row in enumerate(results.itertuples(), 1):
        st.write(f"{i}. **{row.title}** — Predicted rating: {row.predicted_rating:.2f}⭐")
