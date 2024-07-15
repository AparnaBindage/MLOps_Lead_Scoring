'''
filename: utils.py
functions: encode_features, get_train_model
creator: shashank.gupta
version: 1
'''

###############################################################################
# Import necessary modules
# ##############################################################################

import pandas as pd
import numpy as np

import sqlite3
from sqlite3 import Error

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Lead_scoring_training_pipeline.constants import *


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from datetime import datetime
from datetime import date

###############################################################################
# Define the function to encode features
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
       

    OUTPUT
        1. Save the encoded features in a table - features
        2. Save the target variable in a separate table - target


    SAMPLE USAGE
        encode_features()
        
    **NOTE : You can modify the encode_featues function used in heart disease's inference
        pipeline from the pre-requisite module for this.
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
            
    remaining_df = df[['city_tier', 'total_leads_droppped', 'referred_lead', 'app_complete_flag']]
    encoded_df = pd.concat([placeholder_df, remaining_df], axis=1)
    
    # fill all null values
    encoded_df.fillna(0, inplace=True)

    features = encoded_df.drop('app_complete_flag', axis=1)
    target = encoded_df[['app_complete_flag']]
   
    features.to_sql(name='features', con=cnx,if_exists='replace',index=False)
    target.to_sql(name='target', con=cnx,if_exists='replace',index=False)


###############################################################################
# Define the function to train the model
# ##############################################################################

def get_trained_model():
    '''
    This function setups mlflow experiment to track the run of the training pipeline. It 
    also trains the model based on the features created in the previous function and 
    logs the train model into mlflow model registry for prediction. The input dataset is split
    into train and test data and the auc score calculated on the test data and
    recorded as a metric in mlflow run.   

    INPUTS
        db_file_name : Name of the database file
        db_path : path where the db file should be


    OUTPUT
        Tracks the run in experiment named 'Lead_scoring_mlflow_production'
        Logs the trained model into mlflow model registry with name 'LightGBM'
        Logs the metrics and parameters into mlflow run
        Calculate auc from the test data and log into mlflow run  

    SAMPLE USAGE
        get_trained_model()
    '''
    """ create a database connection to a SQLite database """
   
    conn = None
    # opening the conncetion for creating the sqlite db
    try:
        conn = sqlite3.connect(DB_FILE_MLFLOW)
        print(sqlite3.version)
    # return an error if connection not established
    except Error as e:
        print(e)
    # closing the connection once the database is created
    finally:
        if conn:
             conn.close()
    mlflow.set_tracking_uri(TRACKING_URI)
    
    cnx = sqlite3.connect(DB_PATH+DB_FILE_NAME)
    X = pd.read_sql('select * from features', cnx)
    y = pd.read_sql('select * from target', cnx)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
   

    #Model Training

    with mlflow.start_run(run_name='run_LightGB_model') as run:
        #Model Training
        clf = lgb.LGBMClassifier()
        clf.set_params(**model_config) 
        clf.fit(X_train, y_train)

        mlflow.sklearn.log_model(sk_model=clf,artifact_path="models", registered_model_name='LightGBM')
        mlflow.log_params(model_config)    

        # predict the results on test dataset
        y_pred=clf.predict(X_test)

        #Log metrics
        acc=accuracy_score(y_pred, y_test)
        conf_mat = confusion_matrix(y_pred, y_test)
        precision = precision_score(y_pred, y_test,average= 'macro')
        recall = recall_score(y_pred, y_test, average= 'macro')
        f1 = f1_score(y_pred, y_test, average='macro')
        cm = confusion_matrix(y_test, y_pred)
        auc = roc_auc_score(y_pred, y_test)
        tn = cm[0][0]
        fn = cm[1][0]
        tp = cm[1][1]
        fp = cm[0][1]
        class_zero = precision_recall_fscore_support(y_test, y_pred, average='binary',pos_label=0)
        class_one = precision_recall_fscore_support(y_test, y_pred, average='binary',pos_label=1)

        mlflow.log_metric('AUC', auc)
        mlflow.log_metric('test_accuracy', acc)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("Precision_0", class_zero[0])
        mlflow.log_metric("Precision_1", class_one[0])
        mlflow.log_metric("Recall_0", class_zero[1])
        mlflow.log_metric("Recall_1", class_one[1])
        mlflow.log_metric("f1_0", class_zero[2])
        mlflow.log_metric("f1_1", class_one[2])
        mlflow.log_metric("False Negative", fn)
        mlflow.log_metric("True Negative", tn)

        runID = run.info.run_uuid
        print("Inside MLflow Run with id {}".format(runID))