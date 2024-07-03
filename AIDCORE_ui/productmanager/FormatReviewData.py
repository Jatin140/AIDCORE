import pandas as pd
import sys
import os


def analyze_gift_sample(gift_file):
    gift_df = pd.read_csv(gift_file)
    print(f"Structure of {gift_file}:")
    print(gift_df.info())
    print(f"\nSample data from {gift_file}:")
    print(gift_df.head())
    return gift_df.columns.tolist()


def create_phone_csv(review_file, item_file, gift_columns, output_file):
    # Read the review and item data
    review_df = pd.read_csv(review_file)
    item_df = pd.read_csv(item_file)

    # Merge review data with item data
    phone_df = pd.merge(review_df, item_df[['asin', 'title', 'brand', 'price']], on='asin', how='left')

    # Rename columns to match giftsample.csv structure
    column_mapping = {
        'asin': 'product_id',
        'name': 'customer_id',
        'rating': 'star_rating',
        'date': 'review_date',
        'title_x': 'review_headline',
        'body': 'review_body',
        'helpfulVotes': 'helpful_votes',
        'title_y': 'product_title'
    }
    phone_df = phone_df.rename(columns=column_mapping)

    # Add missing columns
    phone_df['marketplace'] = 'US'
    phone_df['review_id'] = 'R' + phone_df.index.astype(str).str.zfill(10)
    phone_df['product_category'] = 'Wireless'
    phone_df['product_parent'] = phone_df['product_id']  # Using product_id as parent for simplicity
    phone_df['total_votes'] = phone_df['helpful_votes']
    phone_df['vine'] = 'N'
    phone_df['verified_purchase'] = phone_df['verified'].map({True: 'Y', False: 'N'})

    # Ensure all columns from gift_columns are present, add empty ones if missing
    for col in gift_columns:
        if col not in phone_df.columns:
            phone_df[col] = ''

    # Reorder columns to match giftsample.csv
    phone_df = phone_df[gift_columns]

    # Save to CSV
    phone_df.to_csv(output_file, index=False)
    print(f"\nCreated {output_file} with the structure matching {gift_file}")
    print(f"\nSample data from {output_file}:")
    print(phone_df.head())


def create_asin_master(item_file, output_file):
    item_df = pd.read_csv(item_file)
    asin_master = item_df[['asin', 'brand', 'title', 'price', 'image']]
    asin_master.columns = ['asin', 'brand', 'product_title', 'price', 'image']

    # Save to CSV
    asin_master.to_csv(output_file, index=False)
    print(f"\nCreated {output_file} with columns: asin, brand, product_title, price, image")
    print(f"\nSample data from {output_file}:")
    print(asin_master.head())


def main(gift_file, review_file, item_file, phone_output, asin_output):
    gift_columns = analyze_gift_sample(gift_file)
    create_phone_csv(review_file, item_file, gift_columns, phone_output)
    create_asin_master(item_file, asin_output)


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python script_name.py <gift_file> <review_file> <item_file> <phone_output> <asin_output>")
        sys.exit(1)

    gift_file = sys.argv[1]
    review_file = sys.argv[2]
    item_file = sys.argv[3]
    phone_output = sys.argv[4]
    asin_output = sys.argv[5]

    # Check if input files exist
    for file in [gift_file, review_file, item_file]:
        if not os.path.isfile(file):
            print(f"Error: File '{file}' does not exist.")
            sys.exit(1)

    main(gift_file, review_file, item_file, phone_output, asin_output)