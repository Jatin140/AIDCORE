# Data cleaning and merging
import numpy as np
import pandas as pd
import yaml
import re
from clearml import Task, Dataset, Logger
from clearml import PipelineDecorator

@PipelineDecorator.component(return_values=["config"],cache=True)
def load_config(file_path):
    with open(file_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(f"Error parsing YAML: {exc}")


@PipelineDecorator.component(return_values=["items_df","reviews_df"],cache=True)
def load_dataset(config):
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
    print(items_dataset_path)
    print(reviews_dataset_path)
    items_df = pd.read_csv(items_dataset_path+"/"+config["items_data_file"])
    reviews_df = pd.read_csv(reviews_dataset_path+"/"+config["reviews_data_file"])

    return items_df,reviews_df

@PipelineDecorator.component(return_values=["df_tmp"],cache=True)
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

# @PipelineDecorator.component(return_values=["text"],cache=True)
# def text_cleaning(text):
#     text = re.sub("\\b[A-Z0-9]+-[A-Z0-9]+(-[A-Z0-9]+)?\\b"," ",text) # AA-BB12-ZZ kind of word, no meaning
#     text = re.sub("[^a-zA-Z]"," ",text) # All but alphabets
#     text = re.sub("\\b\w\\b"," ",text) # single char surrounded by space
#     text = re.sub("( )+"," ",text) # Merge multiple space to one space
#     return text

@PipelineDecorator.component(return_values=["df"],cache=True)
def data_cleaning_and_updating_df(df,column=None):
    def text_cleaning(text):
        text = re.sub("\\b[A-Z0-9]+-[A-Z0-9]+(-[A-Z0-9]+)?\\b"," ",text) # AA-BB12-ZZ kind of word, no meaning
        text = re.sub("[^a-zA-Z]"," ",text) # All but alphabets
        text = re.sub("\\b\w\\b"," ",text) # single char surrounded by space
        text = re.sub("( )+"," ",text) # Merge multiple space to one space
        return text    
    df[column] = df[column].apply(lambda x: text_cleaning(x))
    # df[column] = df[column].apply(lambda x: x)
    return df

@PipelineDecorator.component(return_values=["merge_dataset"],cache=True)
def merge_dataset(df1,df2):
    merge_dataset =pd.merge(df1,df2,on='asin',how='inner')
    return merge_dataset

@PipelineDecorator.component(return_values=["df"],cache=True)
def memory_saving(df):
    return df

# Data Validation Low priority

@PipelineDecorator.pipeline(name="Pipeline Experiment",project="capstone_AIDCORE_g7",version="0.1")
def main():
    config_file = "config.yml"
    config = load_config(config_file)
    items,reviews = load_dataset(config)    
    items = data_imputation(items)
    reviews = data_imputation(reviews)
    # _ = text_cleaning("Pass")
    items = data_cleaning_and_updating_df(items,"title")
    reviews = data_cleaning_and_updating_df(reviews,"title")
    reviews = data_cleaning_and_updating_df(reviews,"body")

    merged_df = merge_dataset(items, reviews)
    print(merged_df.shape)
    print(merged_df.head(2))
    memory_saving(merged_df)
    # task = Task.init(project_name="Product Dynamics & overall Sentiment Analysis", task_name="Data Cleaning and Merging")
    # task.upload_data(merged_df, "merged_dataset.csv")
    # logger = Logger()
    # logger.report_scalar("Memory Usage", merged_df.memory_usage().sum() / (1024 * 1024))
    # logger.report_table("Data Summary", merged_df.describe())
    # logger.end()
    # print("Data Cleaning and Merging completed successfully!")
    # print("Memory Usage:", merged_df.memory_usage().sum() / (1024 * 1024), "MB")
    
    return merged_df

if __name__ == '__main__':
    PipelineDecorator.run_locally()
    main()
