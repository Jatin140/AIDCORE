import sys
import os
import time
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import requests
import json

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


def get_topics_from_ollama(reviews, num_topics=10):
    prompt = f"""
    You are an expert in analyzing product reviews and extracting key topics. 
    I have a collection of product reviews, and I need you to identify the {num_topics} most prominent topics discussed in these reviews.
    For each topic, provide a short, descriptive name and list the top 5 keywords associated with that topic.
    Here's a sample of the reviews:

    {' '.join(reviews[:100])}  # Sending a sample of 100 reviews to keep the prompt manageable

    Please format your response as a JSON object with the following structure:
    {{
        "topics": [
            {{
                "name": "Topic Name",
                "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]
            }},
            ...
        ]
    }}
    """

    response = requests.post('http://localhost:11434/api/generate',
                             json={
                                 "model": "mistral",
                                 "prompt": prompt,
                                 "stream": False
                             })

    if response.status_code == 200:
        result = response.json()['response']
        try:
            topics = json.loads(result)
            return topics['topics']
        except json.JSONDecodeError:
            print("Error: Couldn't parse Ollama's response as JSON.")
            return []
    else:
        print(f"Error: Received status code {response.status_code} from Ollama.")
        return []


def assign_topic_to_review(review, topics):
    max_overlap = 0
    assigned_topic = 0
    for i, topic in enumerate(topics):
        overlap = sum(1 for keyword in topic['keywords'] if keyword in review)
        if overlap > max_overlap:
            max_overlap = overlap
            assigned_topic = i
    return assigned_topic


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

    # Get topics from Ollama
    print("Getting topics from Ollama...")
    topics = get_topics_from_ollama(df['processed_text'].tolist())

    if not topics:
        print("Failed to get topics from Ollama. Exiting.")
        return

    # Print topics
    print("\nIdentified Topics:")
    for i, topic in enumerate(topics):
        print(f"Topic {i + 1}: {topic['name']}")
        print(f"Keywords: {', '.join(topic['keywords'])}")
        print()

    # Assign topics to reviews
    print("Assigning topics to reviews...")
    df['topic'] = df['processed_text'].apply(lambda x: assign_topic_to_review(x, topics))
    df['topic_name'] = df['topic'].apply(lambda x: topics[x]['name'])

    # Generate output filename
    input_base = os.path.splitext(input_filename)[0]
    output_filename = f'ollama_topic_modeling_results_{input_base}.csv'

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