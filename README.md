# AIDCORE
AI Driven Customer Operations & Realtime Engagement

## Project Overview

AIDCORE is an intelligent product analytics and customer engagement platform that combines machine learning, natural language processing, and web technologies to provide comprehensive insights into product performance and customer sentiment. The system enables businesses to analyze customer reviews, extract meaningful insights, and engage customers through AI-powered recommendations and conversations.

### What AIDCORE Does

1. **Product Review Analysis**: Processes large-scale product review datasets to extract aspect-level sentiment analysis
2. **AI-Powered Product Discovery**: Provides conversational interface for customers to find products based on their requirements
3. **Trend Analysis**: Offers comprehensive analytics including trend analysis, heatmaps, and clustering for products and brands
4. **Customer Engagement**: Generates personalized marketing emails and recommendations
5. **Document Intelligence**: RAG (Retrieval-Augmented Generation) system for querying product documentation and manuals
6. **Admin Management**: Full-featured admin dashboard for product and user management


## System Architecture & Design

### High-Level Architecture

The system follows a modular, microservices-inspired architecture with clear separation of concerns:

- **Data Layer**: ClearML datasets, local CSV files, SQLite databases, and FAISS vector store
- **Processing Layer**: ClearML pipeline, sentiment analysis models, and LangChain RAG system
- **Application Layer**: Streamlit analytics app, Django REST API, and React frontend
- **External Services**: OpenAI API and ClearML platform

### System Components

#### 1. Data Processing Pipeline (AIDCORE_model/)
- **Purpose**: Orchestrates data ingestion, cleaning, and machine learning workflows
- **Key Files**:
  - pipeline.py: ClearML pipeline for data processing and model orchestration
  - predict.py: ML model components (KNN, BERT, OpenAI-based sentiment analysis)
  - config.yml: Centralized configuration for datasets and model parameters


#### 2. Analytics & Chat Application (AIDCORE_model_app/)
- **Purpose**: Main user interface for analytics, product discovery, and customer engagement
- **Key Features**:
  - Conversational product search using OpenAI GPT-4
  - Interactive data visualization (trends, heatmaps, clustering)
  - RAG-based document querying system
  - Email campaign generation
- **Technology**: Streamlit, pandas, scikit-learn, matplotlib, seaborn

#### 3. REST API Backend (AIDCORE_ui/productmanager/)
- **Purpose**: Provides RESTful API for product and user management
- **Key Features**:
  - JWT-based authentication
  - Product CRUD operations
  - User management with role-based access
  - Mock research analysis endpoints
- **Technology**: Django REST Framework, SQLite

#### 4. Web Frontend (AIDCORE_ui/product-manager-frontend/)
- **Purpose**: Modern web interface for administrators and users
- **Key Features**:
  - Admin dashboard for product management
  - User dashboard for product discovery
  - Authentication and authorization
- **Technology**: React, Material-UI, Axios


## Data Flow Design

### 1. Data Ingestion Flow
ClearML Datasets  Processing Pipeline  Data cleaning & merging  Memory optimization  Sentiment analysis  Streamlit App

### 2. Product Discovery Flow
Customer describes requirements  Chat Interface  OpenAI API  Generate query  SQLite Database search  Analytics Engine  Product recommendations + visualizations

### 3. RAG Document Processing Flow
User uploads PDFs  RAG Interface  LangChain processing  OpenAI embeddings  FAISS vector store  User asks questions  Similarity search  Contextual answers

## Key Design Decisions

### 1. Modular Architecture
- **Separation of Concerns**: Each component has a specific responsibility
- **Independent Deployment**: Components can be deployed and scaled independently
- **Technology Diversity**: Best tool for each job (Streamlit for analytics, Django for API, React for UI)

### 2. Data Management Strategy
- **Hybrid Storage**: ClearML for ML datasets, SQLite for application data, FAISS for vector search
- **Data Pipeline**: Automated data processing with ClearML for reproducibility
- **Caching**: FAISS index for fast document retrieval

### 3. AI/ML Integration
- **Multiple Models**: KNN, BERT, and OpenAI for different use cases
- **RAG Architecture**: Combines retrieval and generation for document Q&A
- **Conversational AI**: GPT-4 for natural product discovery
