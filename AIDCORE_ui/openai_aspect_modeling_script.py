import sys
import os
import time
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from openai import OpenAI
import json
from datetime import datetime

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

VALID_ASPECTS = ['phone', 'price', 'camera', 'battery', 'display', 'design', 'software', 'cpu/gpu', 'memory', 'network']

# Initialize OpenAI client
client = OpenAI()  # Replace with your actual API key

def preprocess_text(text):
    if pd.isna(text):
        return ""
    words = nltk.word_tokenize(str(text).lower())
    tagged_words = nltk.pos_tag(words)
    processed_words = [lemmatizer.lemmatize(word) for word, tag in tagged_words
                       if (tag.startswith('JJ') or tag.startswith('NN')) and word not in stop_words]
    return ' '.join(processed_words)

def get_aspects_from_openai(reviews):
    prompt = f"""
    You are an expert in analyzing product reviews and extracting key aspects of products. 
    I have a collection of product reviews, and I need you to identify the aspects discussed in these reviews.
    You MUST ONLY use the following aspect categories:
    {', '.join(VALID_ASPECTS)}

    For each review, you MUST assign it to one of the above aspects. If a review doesn't clearly fit into one category, assign it to the most relevant aspect based on context.

    For each aspect, provide:
    1. The aspect name (which MUST be one of the categories listed above)
    2. A list of the top 5 keywords or phrases associated with that aspect
    3. The overall sentiment (positive, negative, or neutral) for that aspect based on the reviews

    Guidelines:
    - You MUST use ONLY the aspect categories provided above. Do not create or use any other categories.
    - Every review MUST be assigned to one of the given aspects. Do not use "Other" or similar categories.
    - The keywords should reflect the specific terms used in the reviews to discuss that aspect.
    - The sentiment should accurately reflect the opinions in the reviews. Do not default to neutral - use positive or negative when appropriate.

    Here's a sample of the reviews:

    {' '.join(reviews[:100])}  # Sending a sample of 100 reviews to keep the prompt manageable

    Please format your response ONLY as a JSON object with the following structure, without any additional text:
    {{
        "aspects": [
            {{
                "name": "aspect_name",
                "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
                "sentiment": "positive/negative/neutral"
            }},
            ...
        ]
    }}
    Ensure that all aspects from the provided list are included in your response, even if they have neutral sentiment and generic keywords.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # You can change this to "gpt-4" if you have access
            messages=[
                {"role": "system", "content": "You are a product review analysis expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )

        result = response.choices[0].message.content
        print("Raw response from OpenAI:")
        print(result)

        try:
            aspects_data = json.loads(result)
            if 'aspects' in aspects_data:
                return aspects_data['aspects']
            else:
                print("Error: 'aspects' key not found in the JSON response.")
                print("JSON content:")
                print(aspects_data)
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {str(e)}")
            print("Full response:")
            print(result)
    except Exception as e:
        print(f"OpenAI API Error: {str(e)}")

    return []

def assign_aspect_to_review(review, aspects):
    max_overlap = 0
    assigned_aspect = VALID_ASPECTS[0]  # Default to first aspect
    assigned_keywords = []
    assigned_sentiment = 'neutral'
    for aspect in aspects:
        overlap = sum(1 for keyword in aspect['keywords'] if keyword.lower() in review.lower())
        if overlap > max_overlap:
            max_overlap = overlap
            assigned_aspect = aspect['name']
            assigned_keywords = aspect['keywords']
            assigned_sentiment = aspect['sentiment']

    # Determine sentiment based on review content if no clear sentiment was assigned
    if assigned_sentiment == 'neutral':
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'best']
        negative_words = ['bad', 'poor', 'terrible', 'awful', 'hate', 'worst']
        positive_count = sum(1 for word in positive_words if word in review.lower())
        negative_count = sum(1 for word in negative_words if word in review.lower())
        if positive_count > negative_count:
            assigned_sentiment = 'positive'
        elif negative_count > positive_count:
            assigned_sentiment = 'negative'

    return assigned_aspect, assigned_keywords, assigned_sentiment

def main():
    # Get the input filename from the user
    input_filename = input("Please enter the name of your input TSV file (including .tsv extension): ")

    # Start overall timing
    overall_start_time = time.time()

    # Load data into DataFrame
    print("Loading data...")
    load_start_time = time.time()
    df = pd.read_csv(input_filename, sep='\t', usecols=['review_body', 'star_rating', 'product_id', 'product_title'])
    df = df.dropna(subset=['review_body', 'product_id', 'product_title']).reset_index(drop=True)
    load_end_time = time.time()
    print(f"Data loaded and cleaned. Shape: {df.shape}")
    print(f"Time taken to load data: {load_end_time - load_start_time:.2f} seconds")

    # Process all text data at once
    print("Processing text data...")
    preprocess_start_time = time.time()
    df['processed_text'] = df['review_body'].apply(preprocess_text)
    preprocess_end_time = time.time()
    print(f"Time taken to preprocess text: {preprocess_end_time - preprocess_start_time:.2f} seconds")

    # Get aspects from OpenAI
    print("Getting aspects from OpenAI...")
    openai_start_time = time.time()
    aspects = get_aspects_from_openai(df['processed_text'].tolist())
    openai_end_time = time.time()
    print(f"Time taken to get aspects from OpenAI: {openai_end_time - openai_start_time:.2f} seconds")

    if not aspects:
        print("Failed to get aspects from OpenAI. Exiting.")
        return

    # Print aspects
    print("\nIdentified Aspects:")
    for i, aspect in enumerate(aspects):
        print(f"Aspect: {aspect['name']}")
        print(f"Keywords: {', '.join(aspect['keywords'])}")
        print(f"Sentiment: {aspect['sentiment']}")
        print()

    print("Assigning aspects to reviews...")
    assign_start_time = time.time()
    df['aspect'], df['keywords'], df['sentiment'] = zip(
        *df['processed_text'].apply(lambda x: assign_aspect_to_review(x, aspects)))
    df['keywords'] = df['keywords'].apply(
        lambda x: ', '.join(x) if x else ', '.join(aspects[0]['keywords']))  # Use default keywords if none assigned
    assign_end_time = time.time()
    print(f"Time taken to assign aspects: {assign_end_time - assign_start_time:.2f} seconds")

    # Generate output filename with timestamp
    input_base = os.path.splitext(input_filename)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f'openai_aspect_modeling_results_{input_base}_{timestamp}.csv'

    # Save results to CSV
    print("Saving results...")
    save_start_time = time.time()
    df[['review_body', 'star_rating', 'product_id', 'product_title', 'aspect', 'keywords', 'sentiment']].to_csv(output_filename, index=False)
    save_end_time = time.time()
    print(f"Time taken to save results: {save_end_time - save_start_time:.2f} seconds")

    # Overall execution time
    overall_end_time = time.time()
    print(f"\nTotal execution time: {overall_end_time - overall_start_time:.2f} seconds")

    # Display a sample of the results
    print("\nSample results:")
    print(df[['review_body', 'star_rating', 'product_id', 'product_title', 'aspect', 'keywords', 'sentiment']].head())

    # Print sentiment distribution
    print("\nSentiment Distribution:")
    sentiment_counts = df['sentiment'].value_counts()
    for sentiment, count in sentiment_counts.items():
        print(f"{sentiment}: {count} ({count/len(df)*100:.2f}%)")

if __name__ == "__main__":
    main()