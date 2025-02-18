import logging
import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------------------
# Utility Functions
# -------------------------------

@st.cache_data(show_spinner=False)
def load_and_process_data():
    logging.info("Starting data loading process.")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    logging.debug("Loading dataset from Hugging Face.")
    status_text.text("Loading dataset...")
    dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_meta_Electronics",
        trust_remote_code=True,
        split="train[:20%]"
    )
    df = pd.DataFrame(dataset)
    
    logging.debug("Filtering out rows with missing price values.")
    df = df[df['price'].notna()]
    df = df.head(10000)
    
    progress_bar.progress(30)
    time.sleep(0.5)
    
    logging.debug("Filtering relevant columns and handling missing descriptions.")
    df = df[['title', 'description', 'price']]
    df = df[df['description'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
    df.reset_index(drop=True, inplace=True)
    df['joined_description'] = df['description'].apply(lambda x: ' '.join(x))
    
    progress_bar.progress(60)
    time.sleep(0.5)
    
    logging.debug("Performing TF-IDF vectorization.")
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['joined_description'])
    df['tfidf_vector'] = list(tfidf_matrix)
    
    logging.debug("Applying K-Means clustering.")
    num_clusters = 10
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(tfidf_matrix)
    
    progress_bar.progress(80)
    time.sleep(0.5)
    
    logging.debug("Computing SBERT embeddings.")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    with st.spinner("Computing SBERT embeddings..."):
        df['sbert_embedding'] = df['joined_description'].apply(lambda x: sbert_model.encode(x))
    
    progress_bar.progress(100)
    time.sleep(0.5)
    logging.info("Data loading process completed.")
    
    return df, tfidf_vectorizer, kmeans, sbert_model

def search_products(query, df, top_n=5):
    logging.info(f"Searching for products matching: {query}")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df['joined_description'])
    query_vec = vectorizer.transform([query])
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[-top_n:][::-1]
    logging.debug(f"Top search results: {df.iloc[top_indices]['title'].tolist()}")
    return df.iloc[top_indices]

def traditional_recommendations(selected_title, df, top_n=5):
    logging.info(f"Generating TF-IDF recommendations for: {selected_title}")
    try:
        product_cluster = df.loc[df['title'] == selected_title, 'cluster'].values[0]
    except IndexError:
        logging.warning(f"Product not found: {selected_title}")
        return pd.DataFrame()
    rec_df = df[(df['cluster'] == product_cluster) & (df['title'] != selected_title)]
    return rec_df.head(top_n)

def sbert_recommendations(selected_title, df, sbert_model, top_n=5):
    logging.info(f"Generating SBERT recommendations for: {selected_title}")
    try:
        idx = df.index[df['title'] == selected_title][0]
    except IndexError:
        logging.warning(f"Product not found: {selected_title}")
        return pd.DataFrame()
    query_embedding = df.loc[idx, 'sbert_embedding'].reshape(1, -1)
    embeddings = np.vstack(df['sbert_embedding'].values)
    sim_scores = cosine_similarity(query_embedding, embeddings).flatten()
    sim_scores[idx] = -np.inf
    top_indices = sim_scores.argsort()[-top_n:][::-1]
    logging.debug(f"Top SBERT recommendations: {df.iloc[top_indices]['title'].tolist()}")
    return df.iloc[top_indices]

def safe_convert_price(price):
    try:
        return float(price) if price not in [None, "", "None"] else 0.0
    except ValueError:
        logging.error(f"Invalid price encountered: {price}")
        return 0.0

def main():
    st.title("Product Recommendation Engine")
    df, tfidf_vectorizer, kmeans, sbert_model = load_and_process_data()
    
    st.sidebar.title("Settings")
    model_choice = st.sidebar.radio("Select Recommendation Model", ["TF-IDF", "SBERT"])
    
    query = st.text_input("Search for a product by description")
    if query:
        results = search_products(query, df)
        selected_product = st.selectbox("Select a product", results['title'].tolist())
        
        if st.button("Add to Cart"):
            st.session_state.cart.append(selected_product)
            st.session_state.total_spend += safe_convert_price(df[df['title'] == selected_product]['price'].values[0])
    
    if st.session_state.cart:
        st.write("### Cart")
        st.write(st.session_state.cart)
        st.write(f"Total Spend: ${st.session_state.total_spend:.2f}")
        
        if model_choice == "TF-IDF":
            recommendations = traditional_recommendations(st.session_state.cart[-1], df)
        else:
            recommendations = sbert_recommendations(st.session_state.cart[-1], df, sbert_model)
        
        st.write("### Recommendations")
        selected_recommendation = st.selectbox("Select a recommended product", recommendations['title'].tolist())
        if st.button("Add Recommended to Cart"):
            st.session_state.cart.append(selected_recommendation)
            st.session_state.total_spend += safe_convert_price(df[df['title'] == selected_recommendation]['price'].values[0])

if __name__ == "__main__":
    if 'cart' not in st.session_state:
        st.session_state.cart = []
        st.session_state.total_spend = 0.0
    main()