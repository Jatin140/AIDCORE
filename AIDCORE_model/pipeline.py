# Data cleaning and merging
import numpy as np
import pandas as pd
import yaml
import time
import re,sys
import subprocess
import traceback
from clearml import Task, Dataset, Logger
from clearml import PipelineDecorator, PipelineController
import predict


# Add the path of othe rmodules so that library can be imported
sys.path.append('../')

from AIDCORE_model import *
from AIDCORE_utils.send_email import send_email_to_product_owner

@PipelineDecorator.component(return_values=["config"],cache=False)
def load_config(file_path):
    logger = PipelineController.get_logger()
    logger.report_text("Loading config...")

    with open(file_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(f"Error parsing YAML: {exc}")


@PipelineDecorator.component(return_values=["items_df","reviews_df"],cache=False)
def load_dataset(config):
    logger = PipelineController.get_logger()
    logger.report_text("Loading items.csv...")

    items_dataset_path = Dataset.get(
            dataset_id=config["items_data_file_id"],
            only_completed=True, 
            only_published=False,
            alias='latest', 
    ).get_local_copy()

    logger.report_text("Loading reviws.csv...")
    reviews_dataset_path = Dataset.get(
            dataset_id=config["reviews_data_file_id"],
            only_completed=True,
            only_published=False,
            alias='latest',
    ).get_local_copy()
    print(items_dataset_path)
    print(reviews_dataset_path)
    items_df = pd.read_csv(items_dataset_path+"/"+config["items_data_file"])
    reviews_df = pd.read_csv(reviews_dataset_path+"/"+config["reviews_data_file"])

    return items_df,reviews_df

@PipelineDecorator.component(return_values=["df_tmp"],cache=False)
def data_imputation(df):
    logger = PipelineController.get_logger()
    logger.report_text("Data Imputation...")    
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

# @PipelineDecorator.component(return_values=["text"],cache=False)
# def text_cleaning(text):
#     text = re.sub("\\b[A-Z0-9]+-[A-Z0-9]+(-[A-Z0-9]+)?\\b"," ",text) # AA-BB12-ZZ kind of word, no meaning
#     text = re.sub("[^a-zA-Z]"," ",text) # All but alphabets
#     text = re.sub("\\b\w\\b"," ",text) # single char surrounded by space
#     text = re.sub("( )+"," ",text) # Merge multiple space to one space
#     return text

@PipelineDecorator.component(return_values=["df"],cache=False)
def data_cleaning_and_updating_df(df,column=None):
    logger = PipelineController.get_logger()
    logger.report_text("Data Cleaning...")    
    def text_cleaning(text):
        text = re.sub("\\b[A-Z0-9]+-[A-Z0-9]+(-[A-Z0-9]+)?\\b"," ",text) # AA-BB12-ZZ kind of word, no meaning
        text = re.sub("[^a-zA-Z]"," ",text) # All but alphabets
        text = re.sub("\\b\w\\b"," ",text) # single char surrounded by space
        text = re.sub("( )+"," ",text) # Merge multiple space to one space
        return text    
    df[column] = df[column].apply(lambda x: text_cleaning(x))
    # df[column] = df[column].apply(lambda x: x)
    return df

@PipelineDecorator.component(return_values=["merge_dataset"],cache=False)
def merge_dataset(df1,df2):
    logger = PipelineController.get_logger()
    logger.report_text("Merging Dataset...")    

    merge_dataset =pd.merge(df1,df2,on='asin',how='inner')
    merge_dataset.drop(["helpfulVotes"],axis=1,inplace=True)
    merge_dataset.dropna(inplace=True)
    print(merge_dataset.shape)
    return merge_dataset

@PipelineDecorator.component(return_values=["df"],cache=False)
def memory_saving(df):
    """
    Downgrade integer and float columns in a DataFrame to the smallest possible type
    based on the range of values in each column.
    """
    logger = PipelineController.get_logger()
    logger.report_text("Optimizing memory occupied by data...")    

    # Downgrade integer columns
    for col in df.select_dtypes(include=['int64']).columns:
        min_val, max_val = df[col].min(), df[col].max()
        if np.iinfo(np.int8).min <= min_val <= np.iinfo(np.int8).max and np.iinfo(np.int8).min <= max_val <= np.iinfo(np.int8).max:
            df[col] = df[col].astype(np.int8)
        elif np.iinfo(np.int16).min <= min_val <= np.iinfo(np.int16).max and np.iinfo(np.int16).min <= max_val <= np.iinfo(np.int16).max:
            df[col] = df[col].astype(np.int16)
    
    # Downgrade float columns
    for col in df.select_dtypes(include=['float64']).columns:
        min_val, max_val = df[col].min(), df[col].max()
        if np.finfo(np.float32).min <= min_val <= np.finfo(np.float32).max and np.finfo(np.float32).min <= max_val <= np.finfo(np.float32).max:
            df[col] = df[col].astype(np.float32)

    return df

@PipelineDecorator.component(return_values=["None"],cache=False)
def launch_AIDCORE_app(df):
    """
    Launches the AIDCORE Streamlit app.
    """
    command = ["streamlit", "run", "../AIDCORE_model_app/main_app.py"]

    # # Run the command using subprocess.Popen for better control
    process = subprocess.Popen(command)

    try:
        while True:
            # Check if the Streamlit app process has exited
            retcode = process.poll()
            if retcode is not None:
                # The Streamlit app has exited, break the loop
                print(f"Streamlit app exited with code {retcode}")
                break
            # Keep the launcher script running to keep the app alive
            time.sleep(1)     
    except KeyboardInterrupt:
        print("\nStreamlit app interrupted by user. Exiting gracefully...")
        process.terminate()  # Terminate the subprocess
        process.wait()
    
    return retcode

@PipelineDecorator.component(return_values=["None"],cache=False)
def add_latestFeedback_to_main_data(review_feedback,main_data):
    """
    TBD
    """
    logger = PipelineController.get_logger()
    logger.report_text("Adding latest review feedback to Original dataset...")  

    return None

@PipelineDecorator.component(return_values=["None"],cache=False)
def update_DataSet_to_Server(updated_dataset):
    """
    TBD
    """
    logger = PipelineController.get_logger()
    logger.report_text("Updating Original dataset in Server...")  

    return None

@PipelineDecorator.pipeline(name="Pipeline Experiment",project="capstone_AIDCORE_g7",version="1.0")
def main():
    logger = PipelineController.get_logger()

    config_file = "./config.yml"
    config = load_config(config_file)
    items,reviews_main = load_dataset(config)    
    items = data_imputation(items)
    reviews = data_imputation(reviews_main)
    items = data_cleaning_and_updating_df(items,"title")
    reviews = data_cleaning_and_updating_df(reviews,"title")
    reviews = data_cleaning_and_updating_df(reviews,"body")
    merged_df = merge_dataset(items, reviews)
    print(merged_df.shape)
    print(merged_df.head(2))
    memory_optimized_data = memory_saving(merged_df)
    memory_optimized_data.to_csv("./memory_optimized_file.csv")

    # Predictions & Model evaluation
    knn_df = predict.genrate_SentimentAspectFrom_KNN(memory_optimized_data)
    bert_df = predict.genrate_SentimentAspectFrom_BERT(memory_optimized_data)
    knn_metrics = predict.evaluate_KNN_model(knn_df)
    bert_metrics = predict.evaluate_BERT_model(bert_df)
    openai_metrics = "TBD"
    final_metrics = predict.compare_all_models(openai_metrics,knn_metrics,bert_metrics)
    logger.report_text("final_metrics is\n {}...".format(final_metrics))    

    scalar_series = np.random.randint(0,10)

    logger.report_scalar(title="Memory Usage",series="series",iteration=scalar_series,value=merged_df.memory_usage().sum() / (1024 * 1024))
    logger.report_table(title="Data Summary",series="series",iteration=scalar_series,table_plot=merged_df.describe())
    print("Data Cleaning and Merging completed successfully!")
    print("Memory Usage:", merged_df.memory_usage().sum() / (1024 * 1024), "MB")

    # Serving model using app and launching streamlit app
    return_code = launch_AIDCORE_app(final_metrics)
    
    if return_code == 0:
        updated_dataset = add_latestFeedback_to_main_data(reviews_main,None)
        _ = update_DataSet_to_Server(updated_dataset)
    
        # Send an email to product owner in case any negative reviews logged in by user -->TBD
        send_email_to_product_owner(_)
        exit(return_code)

    

if __name__ == '__main__':
    PipelineDecorator.run_locally()
    main()
