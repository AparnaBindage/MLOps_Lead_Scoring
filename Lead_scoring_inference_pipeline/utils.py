'''
filename: utils.py
functions: encode_features, load_model
creator: shashank.gupta
version: 1
'''

###############################################################################
# Import necessary modules
# ##############################################################################

import mlflow
import mlflow.sklearn
import pandas as pd

import sqlite3

import os
import logging
import time
from datetime import datetime
from Lead_scoring_inference_pipeline.constants import *

###############################################################################
# Define the function to train the model
# ##############################################################################


def encode_features():
    '''
    This function one hot encodes the categorical features present in our  
    training dataset. This encoding is needed for feeding categorical data 
    to many scikit-learn models.

    INPUTS
        db_file_name : Name of the database file 
        db_path : path where the db file should be
        ONE_HOT_ENCODED_FEATURES : list of the features that needs to be there in the final encoded dataframe
        FEATURES_TO_ENCODE: list of features  from cleaned data that need to be one-hot encoded
        **NOTE : You can modify the encode_featues function used in heart disease's inference
        pipeline for this.

    OUTPUT
        1. Save the encoded features in a table - features

    SAMPLE USAGE
        encode_features()
    '''
    cnx = sqlite3.connect(DB_PATH+DB_FILE_NAME)
    df = pd.read_sql('select * from model_input', cnx)
   
    # Implement these steps to prevent dimension mismatch during inference
    encoded_df = pd.DataFrame(columns= ONE_HOT_ENCODED_FEATURES) # from constants.py
    placeholder_df = pd.DataFrame()
    
    # One-Hot Encoding using get_dummies for the specified categorical features
    for f in FEATURES_TO_ENCODE:
        if(f in df.columns):
            print("In encoding")
            encoded = pd.get_dummies(df[f])
            encoded = encoded.add_prefix(f + '_')
            placeholder_df = pd.concat([placeholder_df, encoded], axis=1)
        else:
            print('Feature not found')
  
    remaining_df = df[['city_tier', 'total_leads_droppped', 'referred_lead']]
    encoded_df = pd.concat([placeholder_df, remaining_df], axis=1)
   
    encoded_df.to_sql(name='features', con=cnx,if_exists='replace',index=False)
    
    

###############################################################################
# Define the function to load the model from mlflow model registry
# ##############################################################################

def get_models_prediction():
    '''
    This function loads the model which is in production from mlflow registry and 
    uses it to do prediction on the input dataset. Please note this function will the load
    the latest version of the model present in the production stage. 

    INPUTS
        db_file_name : Name of the database file
        db_path : path where the db file should be
        model from mlflow model registry
        model name: name of the model to be loaded
        stage: stage from which the model needs to be loaded i.e. production


    OUTPUT
        Store the predicted values along with input data into a table

    SAMPLE USAGE
        load_model()
    '''
    
    mlflow.set_tracking_uri("http://0.0.0.0:6006")
    cnx = sqlite3.connect(DB_PATH+DB_FILE_NAME)  #db path
 
    logged_model = ml_flow_path
    # Load model as a PyFuncModel.
    loaded_model = mlflow.sklearn.load_model(logged_model)
    # Predict on a Pandas DataFrame.
    X = pd.read_sql('select * from features', cnx)
    
    predictions_proba = loaded_model.predict_proba(pd.DataFrame(X))
    predictions = loaded_model.predict(pd.DataFrame(X))
    pred_df = X.copy()

    pred_df['app_complete_flag'] = predictions
    
    pred_df[["Prob of Not completing application","Prob of completing application"]] = predictions_proba
    print(pred_df.shape)
    print(pred_df.head())
    pred_df.to_sql(name='predictions', con=cnx,if_exists='replace',index=False)

###############################################################################
# Define the function to check the distribution of output column
# ##############################################################################

def prediction_ratio_check():
    '''
    This function calculates the % of 1 and 0 predicted by the model and  
    and writes it to a file named 'prediction_distribution.txt'.This file 
    should be created in the ~/airflow/dags/Lead_scoring_inference_pipeline 
    folder. 
    This helps us to monitor if there is any drift observed in the predictions 
    from our model at an overall level. This would determine our decision on 
    when to retrain our model.
    

    INPUTS
        db_file_name : Name of the database file
        db_path : path where the db file should be

    OUTPUT
        Write the output of the monitoring check in prediction_distribution.txt with 
        timestamp.

    SAMPLE USAGE
        prediction_col_check()
    '''
    cnx = sqlite3.connect(DB_PATH+DB_FILE_NAME)  
    df = pd.read_sql('select * from predictions', cnx)
  
    count_of_1s = df['app_complete_flag'].value_counts()[1]
    count_of_0s = df['app_complete_flag'].value_counts()[0]
    total_count = count_of_0s+count_of_1s
 
    timestamp = str(int(time.time()))
    f = open(FILE_PATH, "a")
    f.write("Prediction at timestamp {} - \nPercentage of 1s : {} \nPercentage of 0s : {} \n".format(timestamp, (count_of_1s/total_count) * 100, (count_of_0s/total_count) * 100))
    f.close()
    
    
###############################################################################
# Define the function to check the columns of input features
# ##############################################################################


def input_features_check():
    '''
    This function checks whether all the input columns are present in our new
    data. This ensures the prediction pipeline doesn't break because of change in
    columns in input data.

    INPUTS
        db_file_name : Name of the database file
        db_path : path where the db file should be
        ONE_HOT_ENCODED_FEATURES: List of all the features which need to be present
        in our input data.

    OUTPUT
        It writes the output in a log file based on whether all the columns are present
        or not.
        1. If all the input columns are present then it logs - 'All the models input are present'
        2. Else it logs 'Some of the models inputs are missing'

    SAMPLE USAGE
        input_col_check()
    '''
    flag = 0
    cnx = sqlite3.connect(DB_PATH+DB_FILE_NAME)  
    df = pd.read_sql('select * from features', cnx)
  
    for feature in ONE_HOT_ENCODED_FEATURES:
        if feature not in df.columns:
            flag = 1
            print('Some of the models inputs are missing ')
            df[feature] = 0
    if flag == 0:
        print('All the models input are present')
    df.to_sql(name='features', con=cnx,if_exists='replace',index=False)