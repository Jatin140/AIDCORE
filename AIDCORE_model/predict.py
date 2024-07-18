# Data cleaning and merging
import numpy as np
import pandas as pd
import yaml
import re
from clearml import PipelineDecorator, PipelineController

@PipelineDecorator.component(return_values=["df"],cache=False)
def genrate_SentimentAspectFrom_KNN(df):
    """
    TBD
    """
    logger = PipelineController.get_logger()
    logger.report_text("Getting Aspect and Sentiments from KNN...")    

    return df

@PipelineDecorator.component(return_values=["df"],cache=False)
def genrate_SentimentAspectFrom_BERT(df):
    """
    TBD
    """
    logger = PipelineController.get_logger()
    logger.report_text("Getting Aspect and Sentiments from BERT...")        
    
    return df

@PipelineDecorator.component(return_values=["knn_metrics"],cache=False)
def evaluate_KNN_model(df):
    """
    TBD
    """
    logger = PipelineController.get_logger()
    logger.report_text("Evaluate KNN over OpenAI...")        
    
    knn_metrics = "TBD"
    return knn_metrics

@PipelineDecorator.component(return_values=["bert_metrics"],cache=False)
def evaluate_BERT_model(df):
    """
    TBD
    """
    logger = PipelineController.get_logger()
    logger.report_text("Evaluate BERT over OpenAI...")        

    bert_metrics = "TBD"
    return bert_metrics


@PipelineDecorator.component(return_values=["final_metrics"],cache=False)
def compare_all_models(openai_metrics,knn_metrics,bert_metrics):
    """
    TBD
    """
    logger = PipelineController.get_logger()
    logger.report_text("Compare all models and report final statistics...")    

    final_metrics = "TBD" 
    
    return final_metrics

