import sys
import os
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from gensim.models import Word2Vec
import multiprocessing
from functools import partial

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()


def extract_adjectives(text):
    if pd.isna(text):
        return ""
    words = nltk.word_tokenize(str(text).lower())
    tagged_words = nltk.pos_tag(words)
    adjectives = [lemmatizer.lemmatize(word) for word, tag in tagged_words if tag.startswith('JJ')]
    return ' '.join(adjectives)


def process_chunk(chunk, tfidf_model, km):
    chunk = chunk.copy()
    chunk['processed_text'] = chunk['review_body'].apply(extract_adjectives)
    tfidf_matrix = tfidf_model.transform(chunk['processed_text'])
    chunk['cluster'] = km.predict(tfidf_matrix)
    return chunk[['review_body', 'star_rating', 'product_id', 'product_title', 'cluster', 'processed_text']]


def main():
    # Get the input filename from the user
    input_filename = input("Please enter the name of your input TSV file (including .tsv extension): ")

    # Start overall timing
    overall_start_time = time.time()

    # Load data into DataFrame
    print("Loading data...")
    df = pd.read_csv(input_filename, sep='\t', usecols=['review_body', 'star_rating', 'product_id', 'product_title'])
    df = df.dropna(subset=['review_body', 'product_id', 'product_title']).reset_index(drop=True)
    print(f"Data loaded and cleaned. Shape: {df.shape}")

    # Process all text data at once
    print("Processing text data...")
    df['processed_text'] = df['review_body'].apply(extract_adjectives)

    # Initialize and fit TF-IDF model
    print("Fitting TF-IDF model...")
    tfidf_model = TfidfVectorizer(max_df=0.99, max_features=1000, min_df=0.01, stop_words='english', use_idf=True,
                                  ngram_range=(1, 1))
    tfidf_matrix = tfidf_model.fit_transform(df['processed_text'])

    # Perform clustering
    print("Performing clustering...")
    num_clusters = 10
    km = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=1000)
    km.fit(tfidf_matrix)

    # Assign clusters
    df['cluster'] = km.predict(tfidf_matrix)

    # Train Word2Vec model
    print("Training Word2Vec model...")
    all_words = [text.split() for text in df['processed_text'] if text]
    word2vec_model = Word2Vec(all_words, vector_size=100, window=5, min_count=1, workers=multiprocessing.cpu_count())

    # Improved cluster naming function
    def improved_cluster_naming(km, tfidf_model, word2vec_model):
        feature_names = tfidf_model.get_feature_names_out()
        cluster_names = {}

        for i in range(km.n_clusters):
            cluster_center = km.cluster_centers_[i]
            sorted_indices = cluster_center.argsort()[::-1]
            top_feature_indices = sorted_indices[:20]
            top_features = [feature_names[j] for j in top_feature_indices]

            combined_words = set(top_features[:5])
            for word in top_features[:5]:
                try:
                    similar_words = word2vec_model.wv.most_similar(word, topn=1)
                    combined_words.update([similar_word for similar_word, _ in similar_words])
                except KeyError:
                    continue

            cluster_name = ' '.join(list(combined_words)[:5])
            cluster_names[i] = cluster_name

        return cluster_names

    # Find the top terms per cluster and create improved cluster names
    print("Naming clusters...")
    cluster_names = improved_cluster_naming(km, tfidf_model, word2vec_model)
    df['cluster_name'] = df['cluster'].map(cluster_names)

    # Generate output filename
    input_base = os.path.splitext(input_filename)[0]
    output_filename = f'clustering_results_{input_base}.csv'

    # Save results to CSV
    print("Saving results...")
    df[['review_body', 'star_rating', 'product_id', 'product_title', 'cluster', 'cluster_name']].to_csv(output_filename,
                                                                                                        index=False)

    # Overall execution time
    overall_end_time = time.time()
    print(f"Total execution time: {overall_end_time - overall_start_time:.2f} seconds")

    # Display a sample of the results
    print(df[['review_body', 'star_rating', 'product_id', 'product_title', 'cluster', 'cluster_name']].head())


if __name__ == "__main__":
    main()