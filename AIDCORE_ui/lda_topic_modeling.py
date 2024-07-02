import sys
import os
import time
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    if pd.isna(text):
        return ""
    words = nltk.word_tokenize(str(text).lower())
    tagged_words = nltk.pos_tag(words)
    processed_words = [lemmatizer.lemmatize(word) for word, tag in tagged_words
                       if (tag.startswith('JJ') or tag.startswith('NN')) and word not in stop_words]
    return ' '.join(processed_words)


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")


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
    df['processed_text'] = df['review_body'].apply(preprocess_text)

    # Initialize and fit CountVectorizer
    print("Fitting CountVectorizer...")
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english', max_features=1000)
    dtm = vectorizer.fit_transform(df['processed_text'])

    # Perform LDA
    print("Performing LDA...")
    num_topics = 10
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42, max_iter=10)
    lda.fit(dtm)

    # Print top words for each topic
    print("\nTop words for each topic:")
    print_top_words(lda, vectorizer.get_feature_names_out(), n_top_words=10)

    # Assign topics to reviews
    topic_probs = lda.transform(dtm)
    df['topic'] = topic_probs.argmax(axis=1)

    # Create topic names
    feature_names = vectorizer.get_feature_names_out()
    topic_names = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-5 - 1:-1]]
        topic_names.append(', '.join(top_words))

    topic_name_mapping = {i: name for i, name in enumerate(topic_names)}
    df['topic_name'] = df['topic'].map(topic_name_mapping)

    # Generate output filename
    input_base = os.path.splitext(input_filename)[0]
    output_filename = f'topic_modeling_results_{input_base}.csv'

    # Save results to CSV
    print("Saving results...")
    df[['review_body', 'star_rating', 'product_id', 'product_title', 'topic', 'topic_name']].to_csv(output_filename,
                                                                                                    index=False)

    # Overall execution time
    overall_end_time = time.time()
    print(f"\nTotal execution time: {overall_end_time - overall_start_time:.2f} seconds")

    # Display a sample of the results
    print("\nSample results:")
    print(df[['review_body', 'star_rating', 'product_id', 'product_title', 'topic', 'topic_name']].head())


if __name__ == "__main__":
    main()