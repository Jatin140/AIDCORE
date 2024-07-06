# Data cleaning and merging
import numpy as np
import pandas as pd
import yaml
import re
from clearml import Task, Dataset, Logger

def load_config(file_path):
    with open(file_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(f"Error parsing YAML: {exc}")

config_file = "config.yml"
config = load_config(config_file)

def load_dataset():
    items_dataset_path = Dataset.get(
            dataset_id=config["items_data_file_id"],  
            only_completed=True, 
            only_published=False, 
    ).get_local_copy()

    reviews_dataset_path = Dataset.get(
            dataset_id=config["reviews_data_file_id"],  
            only_completed=True, 
            only_published=False, 
    ).get_local_copy()

    items_df = pd.read_csv(items_dataset_path+"/"+config["items_data_file"])
    reviews_df = pd.read_csv(reviews_dataset_path+"/"+config["reviews_data_file"])

    return items_df,reviews_df

def data_imputation(df):
    # get the copy of data
    df_tmp = df.copy()

    # get list of object type features
    object_features = df_tmp.select_dtypes(include=['object']).columns.tolist()

    # get list of int/float type features
    float_features = df_tmp.select_dtypes(include=['float64']).columns.tolist()
    int_features = df_tmp.select_dtypes(include=['int64']).columns.tolist()

    # Drop duplicates and null values
    df_tmp.drop_duplicates(inplace=True)
    df_tmp.dropna(inplace=True)

    # replace NaN values mode and mean
    for col in object_features:
        df_tmp[col].fillna(df_tmp[col].mode()[0], inplace=True)

    for col in float_features:
        df_tmp[col].fillna(df_tmp[col].mean(), inplace=True)

    for col in int_features:
        df_tmp[col].fillna(df_tmp[col].mean(), inplace=True)

    # Drop duplicates and null values
    df_tmp.drop_duplicates(inplace=True)
    
    return df_tmp

def data_cleaning(text):
    op = re.sub("\\b[A-Z0-9]+-[A-Z0-9]+(-[A-Z0-9]+)?\\b"," ",text) # AA-BB12-ZZ kind of word, no meaning
    op = re.sub("[^a-zA-Z]"," ",op) # All but alphabets
    op = re.sub("\\b\w\\b"," ",op) # single char surrounded by space
    op = re.sub("( )+"," ",op) # Merge multiple space to one space
    return op



def merge_dataset(df1,df2):
    merge_dataset =pd.merge(df1,df2,on='asin',how='inner')
    return merge_dataset

def memory_saving(df):
    return df

# Data Validation Low priority

if __name__ == '__main__':
    items,reviews = load_dataset()
    
    items = data_imputation(items)
    reviews = data_imputation(reviews)
    
    items['title'] = items['title'].apply(lambda x: data_cleaning(x))
    reviews['title'] = reviews['title'].apply(lambda x: data_cleaning(x))
    reviews['body'] = reviews['body'].apply(lambda x: data_cleaning(x))
    
    merged_df = merge_dataset(items, reviews)
    # print(merged_df.shape)
    # print(merged_df.head(2))
    # memory_saving(merged_df)
    # task = Task.init(project_name="Product Dynamics & overall Sentiment Analysis", task_name="Data Cleaning and Merging")
    # task.upload_data(merged_df, "merged_dataset.csv")
    # logger = Logger()
    # logger.report_scalar("Memory Usage", merged_df.memory_usage().sum() / (1024 * 1024))
    # logger.report_table("Data Summary", merged_df.describe())
    # logger.end()
    # print("Data Cleaning and Merging completed successfully!")
    # print("Memory Usage:", merged_df.memory_usage().sum() / (1024 * 1024), "MB")
