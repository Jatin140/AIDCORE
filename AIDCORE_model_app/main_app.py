from clearml import PipelineDecorator, PipelineController
import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
from openai import OpenAI
import re
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

user_dict = {
    '1': {"username": "Pankaj", "status": "did_not_buy", "requirements": "good camera"},
    '2': {"username": "Jyotirmoy", "status": "did_not_buy", "requirements": "sleek phone"},
    '3': {"username": "Debolina", "status": "did_not_like", "requirements": "Battery_life"},
    '4': {"username": "Puja", "status": "did_not_like", "requirements": "Display"}
}

def get_user_status(user_id):
    return user_dict.get(user_id, {"username": "User", "status": "unknown", "requirements": "general"})

def gen_prompt(user_status):
    return f"""
                You are an expert email generator. Generate a friendly and professional email to the user.
                Take a close look at the input given. And using the information present in the input generate an exciting email.
                Make it funny and memorable.

            Guidelines:

            -- Write email Subject and Email Body both in proper email format.
            -- Don't leave just by writing email subjects/body.
            -- Make the email body exciting. Don't repeat the example given.
            -- Mention the company name 'AIDCORE' always.
            -- Use some phone names instead of recommendation 1, recommendation 2.
            -- Do not repeat the subject again after Best Regards.

            Example:
                For Input:
                    'username': 'Pankaj',
                    'status': 'did_not_buy',
                    'requirements': 'good camera'

                Produce:
                Subject: Explore new recommendations based on your requirement for a good camera phone

                Hello Pankaj,

                We observed that you did not buy the phone. Based on your requirement for a good camera, here are some recommendations:

                - iPhone 12 Pro
                - Google Pixel 5

                Best regards,
                AIDCORE

            Now generate a similar output for the following requirements:
            username: {user_status["username"]}
            status: {user_status['status']}
            requirement: {user_status["requirements"]}
            """

def generate_email_prompt(user_status,client):
    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": gen_prompt(user_status)}
    ])
    return response.choices[0].message.content

def generate_email_content(user_id,client):
    user_status = get_user_status(user_id)
    email = generate_email_prompt(user_status,client)
    return email

# Columns for sentiment analysis
positive_cols = ['positive_about_phone_unlocking', 'positive_about_camera', 'positive_about_os', 'positive_about_memory', 'positive_about_battery', 'positive_about_network',
                 'positive_about_apps', 'positive_about_screen', 'positive_about_price', 'positive_about_purchase_experience']
negative_cols = ['negative_about_phone_unlocking', 'negative_about_camera', 'negative_about_os', 'negative_about_memory', 'negative_about_battery', 'negative_about_network',
                 'negative_about_apps', 'negative_about_screen', 'negative_about_price', 'negative_about_purchase_experience']
aspects = ['phone unlocking', 'camera', 'OS', 'memory', 'battery', 'network', 'apps', 'screen', 'price', 'purchase experience']

# Columns for clustering
aspect_cols = ['negative about OS', 'negative about apps', 'negative about battery', 'negative about brand',
               'negative about camera', 'negative about carrier', 'negative about customer service', 'negative about design',
               'negative about memory', 'negative about network', 'negative about phone unlocking', 'negative about price',
               'negative about purchase experience', 'negative about screen', 'overall negative feedback', 'overall postive feedback',
               'positive about OS', 'positive about apps', 'positive about battery', 'positive about brand', 'positive about camera',
               'positive about carrier', 'positive about customer service', 'positive about design', 'positive about memory',
               'positive about network', 'positive about phone unlocking', 'positive about price', 'positive about purchase experience',
               'positive about screen', 'rating']

cluster_aspect_list = ['negative about OS', 'negative about apps', 'negative about battery', 'negative about camera',
                       'negative about carrier', 'negative about design', 'negative about memory', 'negative about network',
                       'negative about phone unlocking', 'negative about price', 'negative about screen', 'positive about OS',
                       'positive about apps', 'positive about battery', 'positive about camera', 'positive about carrier',
                       'positive about memory', 'positive about network', 'positive about phone unlocking', 'positive about price',
                       'positive about screen']

# Function to get data from SQLite database
def get_data(sql_query="SELECT * FROM products"):
    conn = sqlite3.connect("../AIDCORE_model_app/products.db")
    print("get_data inside", sql_query)
    df = pd.read_sql(sql_query, conn)
    conn.close()
    return df

# Function to get brands from SQLite database
def get_brands():
    df = get_data()
    return df['brand'].unique().tolist()

# Function to get products from SQLite database
def get_products(brand):
    df = get_data()
    products = df[df['brand'] == brand][['asin', 'title']].set_index('asin')['title'].to_dict()
    return products

# Function to get product specifications from SQLite database
def get_product_specs(asin):
    df = get_data()
    specs = df[df['asin'].isin(asin)]
    return specs

# Function to find best match products based on cosine similarity
def find_best_match(query, data):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(data['title'])
    query_vec = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    filtered_indices = [i for i, similarity in enumerate(cosine_similarities) if similarity > 0.05]
    sorted_indices = sorted(filtered_indices, key=lambda i: cosine_similarities[i], reverse=True)
    num_results = min(len(sorted_indices), 5)
    top_indices = sorted_indices[:num_results]
    print(data.iloc[top_indices])
    return data.iloc[top_indices]

def generate_response(conversation,client):
    response = client.chat.completions.create(model="gpt-4",
    messages=conversation,
    max_tokens=800,
    temperature=0.5)
    return response.choices[0].message.content

def get_query(text):
    config_pattern = re.compile(r'DESIRED CONFIGURATION FOR YOU:(.*?)\*\*\*')
    config_match = config_pattern.search(text)
    configuration = config_match.group(1).strip() if config_match else None

    price_rating_pattern = re.compile(r'PRICE (<= \d+).*?RATING (>= \d+)')
    price_rating_match = price_rating_pattern.search(text)
    price_condition = price_rating_match.group(1) if price_rating_match else None
    rating_condition = price_rating_match.group(2) if price_rating_match else None

    sql_query = f"SELECT * FROM products WHERE price {price_condition} AND rating {rating_condition}"
    return configuration, sql_query

def plot_row(df, prod_num):
    positive_values = df[positive_cols].values[0]
    negative_values = df[negative_cols].values[0]
    df_asp = pd.DataFrame({
        'positive': positive_values,
        'negative': negative_values
    }, index=aspects)
    fig, ax = plt.subplots(figsize=(10, 4))
    df_asp.sort_values('positive').plot(kind='bar', ax=ax)
    ax.set_title(f'Asin: {prod_num[0]}')
    ax.set_xlabel('Aspects')
    ax.set_ylabel('Scores')
    return fig

def plot_overall(df, asins):
    df = df[['overall_neutral', 'overall_positive', 'overall_negative']].sort_values('overall_negative')
    df.index = asins
    fig, ax = plt.subplots(figsize=(10, 4))
    df.plot(kind='barh', ax=ax)
    ax.set_title(f'Overall Review Comparison')
    ax.set_xlabel('Percentage Reviewed')
    return fig

def start_conversation():
    delimiter = "####"
    system_message = f"""
    You are an intelligent cell phone expert and your goal is to find the best cell phone for a user.
    You need to ask relevant questions and understand the user profile by analyzing the user's responses.
    Don't make assumptions about what values to plug into functions. Once you get all the details summarize the 
    specification with 'DESIRED CONFIGURATION FOR YOU:'. In the summary also mention the price and rating as greater than equal to or less than equal to or equal to value mentioned by the user 3 asterisks before and after.

    Arrive at the function call in not more than five questions and answers.

    {delimiter} Here is a sample conversation between the user and assistant:
    User: "Hi, I am an editor."
    Assistant: "Great! As an editor, you likely require a cell phone that can handle demanding tasks. Hence, the cell phone should have High-quality camera for photos and videos.Long-lasting battery for extensive use.You would also need a high end display for better visuals and editing. May I know what kind of work do you primarily focus on? Are you more involved in video editing, photo editing, or both? Understanding the specific type of editing work will help me tailor my recommendations accordingly. Let me know if my understanding is correct until now."
    User: "I edit the videos."
    Assistant: "Thank you for providing that information. So your work involves working with graphics, animations, and rendering, which will require high GPU. Do you work with high-resolution media files, such as 4K videos or RAW photos? Understanding your file sizes will help determine the storage capacity and processing power needed."
    User: "Yes, sometimes I work with 4K videos as well."
    Assistant: "Thank you for the information. Could you kindly let me know your budget for the cell phone? This will help me find options that fit within your price range while meeting the specified requirements."
    User: "my max budget is 100 dollars"
    Assistant: "Great! Do you want user rating to be at least a certain number?"
    User: "Yes. It should be at least three point 5"
    Assistant: "Ok. Here is the Specification for your mobile. DESIRED CONFIGURATION FOR YOU:  6GB Amoled 4500 mah and android. *** PRICE <= 100 RATING >= 3.5 ***"
    {delimiter}

    Start with a short welcome message and encourage the user to share their requirements.
    """
    conversation = [{"role": "system", "content": system_message}]
    return conversation

def purchase_page(client):
    st.title("Product Specifications Viewer")
    st.subheader("Chat with our assistant")

    if 'conversation' not in st.session_state:
        st.session_state.conversation = start_conversation()

    user_input = st.text_input("You: ", "")

    if user_input:
        st.session_state.conversation.append({"role": "user", "content": user_input})
        response = generate_response(st.session_state.conversation,client)             
        st.session_state.conversation.append({"role": "assistant", "content": response})

    for msg in st.session_state.conversation:
        if msg['role'] == 'system':
            continue
        st.write(f"{msg['role'].capitalize()}: {msg['content']}")

    user_query = " ".join([msg['content'] for msg in st.session_state.conversation if msg['role'] == 'user'])
    sql_query = "SELECT * FROM products"
    if user_input:
        if response.count("DESIRED CONFIGURATION FOR YOU") > 0:
            user_query, sql_query = get_query(response)
            print(response)
            print("sql query", sql_query)

    df = get_data(sql_query)

    st.title("Find Products")
    if st.button("Search based on conversation"):
        if user_query:
            matched_products = find_best_match(user_query, df)
            if matched_products.shape[0] > 0:
                st.write("Best match products based on your search:")
                st.dataframe(matched_products[['asin', 'title', 'brand', 'price', 'rating']])

                st.write("Price vs Rating")
                plt.figure(figsize=(6, 4))
                plt.scatter(matched_products['price'], matched_products['rating'], alpha=0.7, color='red')
                plt.xlabel('Price')
                plt.ylabel('Rating')
                plt.title('Price vs Rating')
                st.pyplot(plt)

                asins = matched_products['asin'].tolist()
                titles = matched_products['title'].tolist()
                asins_dict = dict(zip(titles, asins))
                st.experimental_set_query_params(**asins_dict)
                user_review_analysis_page()
            else:
                st.write("No matching products found.")
        else:
            st.warning("Please enter a keyword to search.")

def user_review_analysis_page():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("User Review Analysis")
    st.subheader("Overall Feedback From The User")

    asins = tuple(st.experimental_get_query_params().values())
    titles = tuple(st.experimental_get_query_params().keys())

    if asins:
        sql_query = f"SELECT * FROM aspect_summary"
        df = get_data(sql_query).sample(len(asins))

        st.pyplot(plot_overall(df, asins))
        st.subheader("Aspectwise Individual Product Review")

        j = 0
        for i in df.asin:
            st.write("                                                 ")
            st.write("                                                 ")
            st.markdown(f"<span style='color:orange'>{titles[j]}</span>", unsafe_allow_html=True)
            st.pyplot(plot_row(df[df.asin == i], asins))
            j += 1
    else:
        st.write("No products selected for review analysis.")

# Load Data Function
@st.cache_data
def load_data(filepath):
    data = pd.read_csv(filepath)
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data['quarter'] = data['date'].dt.to_period('Q').astype(str)
    data['year'] = data['date'].dt.to_period('Y').astype(str)
    return data

def clean_data(data, columns):
    for col in columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    return data

def product_analysis_page():
    st.title("Product and Brand Analysis Dashboard")

    data = load_data("../AIDCORE_model_app/review_aspects_final_17993.csv")
    item = pd.read_csv('../AIDCORE_model_app/20191226-items.csv')[['asin','brand']]
    data = data.merge(item, on='asin')

    product_list = st.sidebar.multiselect('Select Products:', options=data['asin'].unique(), default=data['asin'].iloc[0])
    brand_list = st.sidebar.multiselect('Select Brands:', options=data['brand'].unique(), default= ['Nokia', 'Samsung'])
    trend_period = st.sidebar.radio("Select Trend Period:", ('Quarterly', 'Annual'))
    aspect_list = st.sidebar.multiselect('Select Aspects for Analysis:', options=aspect_cols, default=['rating', 'overall postive feedback', 'overall negative feedback'])
    analysis_type = st.sidebar.radio("Select Analysis Type:", ('Trend Analysis', 'Heatmap Analysis', 'Cluster Analysis'))

    data = clean_data(data, aspect_list)

    filtered_data_products = data[data['asin'].isin(product_list)]
    filtered_data_brands = data[data['brand'].isin(brand_list)]

    if trend_period == 'Quarterly':
        aggregated_data_products = filtered_data_products.groupby(['asin', 'quarter'])[aspect_list].mean().reset_index()
        aggregated_data_brands = filtered_data_brands.groupby(['brand', 'quarter'])[aspect_list].mean().reset_index().sort_values('quarter')
        aggregated_data_products['period'] = aggregated_data_products['quarter']
        aggregated_data_brands['period'] = aggregated_data_brands['quarter']
    else:
        aggregated_data_products = filtered_data_products.groupby(['asin', 'year'])[aspect_list].mean().reset_index()
        aggregated_data_brands = filtered_data_brands.groupby(['brand', 'year'])[aspect_list].mean().reset_index().sort_values('year')
        aggregated_data_products['period'] = aggregated_data_products['year']
        aggregated_data_brands['period'] = aggregated_data_brands['year']

    for col in aspect_list:
        aggregated_data_products[col] = pd.to_numeric(aggregated_data_products[col], errors='coerce')
        aggregated_data_brands[col] = pd.to_numeric(aggregated_data_brands[col], errors='coerce')

    if analysis_type == 'Trend Analysis':
        for aspect in aspect_list:
            st.subheader(f'Trend Analysis of {aspect} ({trend_period})')
            plt.figure(figsize=(10, 4))
            sns.lineplot(data=aggregated_data_products, x='period', y=aspect, hue='asin', marker='o')
            sns.lineplot(data=aggregated_data_brands, x='period', y=aspect, hue='brand', marker='x', linestyle='--')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(plt)

    elif analysis_type == 'Heatmap Analysis':
        st.subheader('Heatmap Analysis Across Aspects')
        aspects_heat = aspect_list.copy()
        aspects_heat.extend(['overall postive feedback', 'overall negative feedback', 'rating'])
        corr_data = filtered_data_products[aspects_heat].corr()
        sns.heatmap(corr_data, annot=True, cmap='coolwarm')
        st.pyplot()

    elif analysis_type == 'Cluster Analysis':
        cluster_level = st.sidebar.radio("Select Cluster Level:", ('Product Level', 'Brand Level'))

        st.subheader('Cluster Analysis')

        scaler = StandardScaler()

        if cluster_level == 'Product Level':
            normalized_data = scaler.fit_transform(filtered_data_products[cluster_aspect_list])
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(normalized_data)
            filtered_data_products['Cluster'] = clusters

            st.subheader("Cluster Analysis Results (Product Level)")
            st.dataframe(filtered_data_products.head()[['asin', 'rating', 'title', 'Cluster', 'brand']])

            cluster_aspects = {}
            for cluster in range(3):
                cluster_data = filtered_data_products[filtered_data_products['Cluster'] == cluster][cluster_aspect_list]
                mean_aspects = cluster_data.mean().sort_values(ascending=False)
                top_3_aspects = mean_aspects.head(3).index.tolist()
                cluster_aspects[cluster] = top_3_aspects

            st.subheader("Top 3 Aspects per Cluster (Product Level)")
            for cluster, aspects in cluster_aspects.items():
                st.write(f"Cluster {cluster}: {', '.join(aspects)}")

            st.subheader("Cluster Visualization (Product Level)")
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='overall postive feedback', y='overall negative feedback', hue='Cluster', data=filtered_data_products, palette='viridis')
            plt.title("Cluster Visualization based on Positive and Negative Feedback (Product Level)")
            st.pyplot(plt)

        elif cluster_level == 'Brand Level':
            normalized_data = scaler.fit_transform(filtered_data_brands[cluster_aspect_list])
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(normalized_data)
            filtered_data_brands['Cluster'] = clusters

            st.subheader("Cluster Analysis Results (Brand Level)")
            st.dataframe(filtered_data_brands.head()[['asin', 'rating', 'title', 'Cluster', 'brand']])

            cluster_aspects = {}
            for cluster in range(3):
                cluster_data = filtered_data_brands[filtered_data_brands['Cluster'] == cluster][cluster_aspect_list]
                mean_aspects = cluster_data.mean().sort_values(ascending=False)
                top_3_aspects = mean_aspects.head(3).index.tolist()
                cluster_aspects[cluster] = top_3_aspects

            st.subheader("Top 3 Aspects per Cluster (Brand Level)")
            for cluster, aspects in cluster_aspects.items():
                st.write(f"Cluster {cluster}: {', '.join(aspects)}")

            st.subheader("Cluster Visualization (Brand Level)")
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='overall postive feedback', y='overall negative feedback', hue='Cluster', data=filtered_data_brands, palette='viridis')
            plt.title("Cluster Visualization based on Positive and Negative Feedback (Brand Level)")
            st.pyplot(plt)

def campaign_management(client):
    st.title("Campaign Management")
    st.subheader("Generate Emails for Users")

    user_ids = list(user_dict.keys())
    selected_user_id = st.selectbox("Select User ID:", user_ids)
    selected_user_status = get_user_status(selected_user_id)

    st.write(f"Username: {selected_user_status['username']}")
    st.write(f"Status: {selected_user_status['status']}")
    st.write(f"Requirements: {selected_user_status['requirements']}")

    if st.button("Generate Email"):
        email_content = generate_email_content(selected_user_id,client)
        st.subheader("Generated Email")
        st.write(email_content)

def launch_app():
    st.sidebar.title("Navigation")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    page = st.sidebar.selectbox("Select a page", ["Review Based Product Selection", "Product Analysis", "Campaign Management"])

    var_key = st.text_input("Enter the openAI key... ",type="password")
    client = OpenAI(api_key=var_key)

    with open("openAi.key.txt","w") as fh:
        fh.write(var_key)

    if page == "Review Based Product Selection":
        purchase_page(client)
    elif page == "Product Analysis":
        product_analysis_page()
    elif page == "Campaign Management":
        campaign_management(client)

if __name__ == "__main__":
    launch_app()

