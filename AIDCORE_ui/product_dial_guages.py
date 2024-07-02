import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys


def calculate_product_topic_scores(csv_file_path, product_id):
    df = pd.read_csv(csv_file_path)

    # Filter for the specific product
    df_product = df[df['product_id'] == product_id]

    if df_product.empty:
        print(f"No data found for product ID: {product_id}")
        sys.exit(1)

    # Group by topic_name and calculate mean star_rating
    product_topic_scores = df_product.groupby('topic_name')['star_rating'].agg(['mean', 'count']).reset_index()

    # Rename columns for clarity
    product_topic_scores.columns = ['topic_name', 'avg_rating', 'review_count']

    # Round average rating to 2 decimal places
    product_topic_scores['avg_rating'] = product_topic_scores['avg_rating'].round(2)

    # Calculate overall average rating
    overall_score = df_product['star_rating'].mean().round(2)
    overall_count = df_product['star_rating'].count()

    # Add overall score to the dataframe
    overall_df = pd.DataFrame({
        'topic_name': ['Overall'],
        'avg_rating': [overall_score],
        'review_count': [overall_count]
    })

    product_topic_scores = pd.concat([product_topic_scores, overall_df], ignore_index=True)

    return product_topic_scores, df_product['product_title'].iloc[0]


def create_dial_gauge(ax, rating, topic, count):
    # Set up the dial
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)

    # Create the colormap
    cmap = plt.cm.RdYlGn
    norm = plt.Normalize(1, 5)

    # Draw the dial background
    ax.bar(np.deg2rad(np.linspace(0, 180, 100)), [1] * 100, width=np.deg2rad(180 / 99), bottom=1,
           color=cmap(norm(np.linspace(1, 5, 100))))

    # Draw the dial needle
    ax.bar(np.deg2rad(180 * (rating - 1) / 4), 0.95, width=np.deg2rad(2), bottom=1, color='black')

    # Set up the ticks
    ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180]))
    ax.set_xticklabels(['1', '2', '3', '4', '5'])
    ax.tick_params(axis='x', colors='black', labelsize=8)
    ax.set_yticks([])

    # Add the topic name and rating
    ax.text(0, 0, f"{topic}\n{rating}\n({count} reviews)", ha='center', va='center', fontweight='bold')


def create_dial_gauges(data, product_title, output_file):
    n_topics = len(data)
    n_cols = 3
    n_rows = (n_topics + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), subplot_kw=dict(projection='polar'))
    fig.suptitle(f"Ratings for:\n{product_title}", fontsize=16, wrap=True)

    axs_flat = axs.flatten() if n_rows > 1 else axs

    for i, (_, row) in enumerate(data.iterrows()):
        create_dial_gauge(axs_flat[i], row['avg_rating'], row['topic_name'], row['review_count'])

    # Remove any unused subplots
    for i in range(n_topics, n_rows * n_cols):
        fig.delaxes(axs_flat[i])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file)
    plt.close()
    print(f"Saved dial gauges: {output_file}")


def main(csv_file_path, product_id):
    # Calculate product topic scores
    product_topic_scores, product_title = calculate_product_topic_scores(csv_file_path, product_id)

    # Create dial gauges
    output_file = f'product_{product_id}_dial_gauges.png'
    create_dial_gauges(product_topic_scores, product_title, output_file)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <path_to_csv_file> <product_id>")
        sys.exit(1)

    csv_file_path = sys.argv[1]
    product_id = sys.argv[2]
    main(csv_file_path, product_id)