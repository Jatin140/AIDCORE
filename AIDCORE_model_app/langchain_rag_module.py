import re
import time
from io import BytesIO
from typing import Any, Dict, List
import openai
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain(api_key):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(api_key)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def recommendation_logic():
    data = {
        'User': ['User1', 'User1', 'User1', 'User2', 'User2', 'User3', 'User3', 'User3', 'User3'],
        'Item': ['ItemA', 'ItemB', 'ItemC', 'ItemA', 'ItemB', 'ItemA', 'ItemB', 'ItemC', 'ItemD'],
        'Rating': [5, 3, 4, 2, 5, 3, 4, 2, 1]
    }

    df = pd.DataFrame(data)

    # Create a pivot table
    pivot_table = df.pivot_table(index='User', columns='Item', values='Rating')

    # Fill missing values with 0 (or you could use other methods like filling with mean rating)
    pivot_table.fillna(0, inplace=True)

    # Calculate the similarity matrix
    similarity_matrix = np.dot(pivot_table, pivot_table.T)

    # Convert to a DataFrame for better readability
    similarity_df = pd.DataFrame(similarity_matrix, index=pivot_table.index, columns=pivot_table.index)

    print("User-User Similarity Matrix:")
    print(similarity_df)

    # Normalize the pivot table
    normalized_pivot = pivot_table.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)), axis=1)

    # Calculate the cosine similarity matrix
    cosine_similarity_matrix = np.dot(normalized_pivot, normalized_pivot.T)
    cosine_similarity_df = pd.DataFrame(cosine_similarity_matrix, index=pivot_table.index, columns=pivot_table.index)

    print("Cosine Similarity Matrix:")
    print(cosine_similarity_df)

    def recommend_items(user, pivot_table, similarity_df, num_recommendations=3):
        similar_users = similarity_df[user].sort_values(ascending=False).index[1:]
        user_ratings = pivot_table.loc[user]
        recommendations = {}

        for similar_user in similar_users:
            similar_user_ratings = pivot_table.loc[similar_user]
            for item, rating in similar_user_ratings.items():
                if pd.isna(user_ratings[item]):
                    if item not in recommendations:
                        recommendations[item] = rating
                    else:
                        recommendations[item] += rating

        recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return recommendations[:num_recommendations]

    user_to_recommend = 'User1'
    recommendations = recommend_items(user_to_recommend, pivot_table, similarity_df)
    print(f"Recommendations for {user_to_recommend}: {recommendations}")

    # Heatmap of the similarity matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_df, annot=True, cmap='coolwarm')
    plt.title('User-User Similarity Matrix')
    plt.show()

    # Heatmap of the cosine similarity matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cosine_similarity_df, annot=True, cmap='coolwarm')
    plt.title('Cosine Similarity Matrix')
    plt.show()

def gen_prompt(user_status):
    return """
                You are an expert email generator. Generate a friendly and professional email to user.
                Take a close look at the input given. And using the information present in the input generate an exciting email.
                Make it funny and memorable

            guidelines:

            -- Write email Subject and Email Body both proper email format
            -- Don't leave just by writing email subjects/ body
            -- Make the email body exiciting. Dont repeat the example given
            -- Mention Company name 'AIDCORE' always .
            -- use some phone names instead of recomendation 1, recomendation 2
            -- Do not repeat subject again after Best Regards
            -- Please end the message with regards 'AIDCORE'

             Example:
                for Input:
                    username': 'Pankaj',
                    'status': 'did_not_buy',
                    'requirements': 'good camera'

                produce
                Output:
                Subject: Explore new recommendations based on your requirement for a good camera phone

                Hello Pankaj,

                We observed that you did not buy the phone. Based on your requirement for a good camera, here are some recommendations:

                - iPhone 12 Pro
                - Google Pixel 5

                Best regards,
                AIDCORE

            Now generate a similar output for following requirements
            username : {name}
            status: {status}
            requirement : {req}

            """.format(name=user_status["username"], status=user_status['status'], req=user_status["requirements"])

user_dict = {
    '1': {"username": "Pankaj", "status": "did_not_buy", "requirements": "good camera"},
    '2': {"username": "Jyotirmoy", "status": "did_not_buy", "requirements": "sleek phone"},
    '3': {"username": "Debolina", "status": "did_not_like", "requirements": "Battery_life"},
    '4': {"username": "Puja", "status": "did_not_like", "requirements": "Display"}
}

def get_user_status(user_id):
    return user_dict.get(user_id, {"username": "User", "status": "unknown", "requirements": "general"})

def generate_email_prompt(user_status):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        prompt=gen_prompt(user_status),
        max_tokens=250
    )
    return response.choices[0].text.strip()

def generate_email_content(user_id):
    user_status = get_user_status(user_id)
    email = generate_email_prompt(user_status)
    return email



def email_recommendation(user_id):
    email_content = generate_email_content(user_id)
    user_status = get_user_status(user_id)
    #image = generate_image_GPU(user_status['requirements'])

    # Print the email content
    st.write(f"Email for {user_status['username']} ({user_id}):")
    st.write(email_content)
    #st.write("\nGenerated Image:")
    #st.image(image)

def launch_langchain_rag(api_key):
    st.header("Interactive User Feedback Analysis")
    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")
    if user_question :  # Ensure API key and user question are provided
        user_input(user_question, api_key)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
        if api_key: 
            if st.button("Submit & Process", key="process_button"):  # Check if API key is provided before processing
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks, api_key)
                    st.success("Done")
        else:
            st.warning('api_key not found')

        #api_key = st.text_input("Enter your OpenAI API Key:", type="password", key="api_key_input")
        st.subheader("Email Recommendations")
        user_id = st.selectbox("Select User ID", options=list(user_dict.keys()))
        if st.button("Send Recommendation Email", key="email_recommendation_button"):
            email_recommendation(user_id)
    

# if __name__ == "__main__":
#     launch_langchain_rag()
