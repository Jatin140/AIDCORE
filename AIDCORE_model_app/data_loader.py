import sqlite3
import pandas as pd
import re

# Read CSV file into DataFrame
df = pd.read_csv("../AIDCORE_model_app/20191226-items.csv")
df['review_count'] =  df.groupby('asin').rating.transform(lambda x: len(x))

df_aspect = pd.read_csv("asin_leve_aspects_summary.csv")

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect("products.db")
cursor = conn.cursor()

# Drop the existing table if it exists
cursor.execute('DROP TABLE IF EXISTS products')

# Create table
cursor.execute('''
CREATE TABLE IF NOT EXISTS products (
    asin TEXT PRIMARY KEY,
    brand TEXT,
    title TEXT,
    rating REAL,
    totalReviews INTEGER,
    price REAL,
    originalPrice REAL,
    review_count INTEGER
)
''')

# Insert data into table
for index, row in df.iterrows():
    cursor.execute('''
    INSERT OR REPLACE INTO products (asin, brand, title, rating, totalReviews, price, originalPrice, review_count)
    VALUES (?, ?, ?, ?, ?, ?, ?,?)
    ''', (
        row['asin'],
        row['brand'] if pd.notna(row['brand']) else "Realme",
        row['title'],
        row['rating'],
        row['totalReviews'],
        row['price'],
        row['originalPrice'],
        row['review_count']
    ))

# Commit and close connection
conn.commit()
conn.close()

print("Data uploaded successfully!")


##########################  aspect ################################

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect("products.db")
cursor = conn.cursor()

#Drop the existing table if it exists
cursor.execute('DROP TABLE IF EXISTS aspect_summary')

# Create table
cursor.execute('''
CREATE TABLE IF NOT EXISTS aspect_summary (
    asin TEXT PRIMARY KEY,
    positive_about_carrier REAL,
    negative_about_carrier REAL,
    positive_about_phone_unlocking REAL,
    negative_about_phone_unlocking REAL,
    positive_about_design REAL,
    negative_about_design REAL,
    positive_about_camera REAL,
    negative_about_camera REAL,
    positive_about_os REAL,
    negative_about_os REAL,
    positive_about_memory REAL,
    negative_about_memory REAL,
    positive_about_battery REAL,
    negative_about_battery REAL,
    positive_about_network REAL,
    negative_about_network REAL,
    positive_about_apps REAL,
    negative_about_apps REAL,
    positive_about_customer_service REAL,
    negative_about_customer_service REAL,
    positive_about_brand REAL,
    negative_about_brand REAL,
    positive_about_screen REAL,
    negative_about_screen REAL,
    positive_about_price REAL,
    negative_about_price REAL,
    positive_about_purchase_experience REAL,
    negative_about_purchase_experience REAL,
    rating REAL,
    overall_neutral REAL,
    overall_positive REAL,
    overall_negative REAL
)
''')

# Insert data into table
for index, row in df_aspect.iterrows():
    cursor.execute('''
    INSERT OR REPLACE INTO aspect_summary (
        asin, positive_about_carrier, negative_about_carrier, positive_about_phone_unlocking, negative_about_phone_unlocking,
        positive_about_design, negative_about_design, positive_about_camera, negative_about_camera,
        positive_about_os, negative_about_os, positive_about_memory, negative_about_memory,
        positive_about_battery, negative_about_battery, positive_about_network, negative_about_network,
        positive_about_apps, negative_about_apps, positive_about_customer_service, negative_about_customer_service,
        positive_about_brand, negative_about_brand, positive_about_screen, negative_about_screen,
        positive_about_price, negative_about_price, positive_about_purchase_experience, negative_about_purchase_experience,
        rating, overall_neutral, overall_positive, overall_negative
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        row['asin'], row['positive about carrier'], row['negative about carrier'], row['positive about phone unlocking'],
        row['negative about phone unlocking'], row['positive about design'], row['negative about design'], row['positive about camera'],
        row['negative about camera'], row['positive about OS'], row['negative about OS'], row['positive about memory'],
        row['negative about memory'], row['positive about battery'], row['negative about battery'], row['positive about network'],
        row['negative about network'], row['positive about apps'], row['negative about apps'], row['positive about customer service'],
        row['negative about customer service'], row['positive about brand'], row['negative about brand'], row['positive about screen'],
        row['negative about screen'], row['positive about price'], row['negative about price'], row['positive about purchase experience'],
        row['negative about purchase experience'], row['rating'],
        row['overall neutral'], row['overall positive'], row['overall negative']
    ))

# Commit and close connection
conn.commit()
conn.close()

print("Aspect data uploaded successfully!")