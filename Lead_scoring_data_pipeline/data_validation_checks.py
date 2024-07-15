"""
Import necessary modules
############################################################################## 
"""

import pandas as pd
from Lead_scoring_data_pipeline.schema import *
from Lead_scoring_data_pipeline.constants import *
import sqlite3
from sqlite3 import Error
###############################################################################
# Define function to validate raw data's schema
############################################################################### 

def raw_data_schema_check():
    '''
    This function check if all the columns mentioned in schema.py are present in
    leadscoring.csv file or not.

   
    INPUTS
        DATA_DIRECTORY : path of the directory where 'leadscoring.csv' 
                        file is present
        raw_data_schema : schema of raw data in the form oa list/tuple as present 
                          in 'schema.py'

    OUTPUT
        If the schema is in line then prints 
        'Raw datas schema is in line with the schema present in schema.py' 
        else prints
        'Raw datas schema is NOT in line with the schema present in schema.py'

    
    SAMPLE USAGE
        raw_data_schema_check
    '''
    flag = 0
    loaded_data = pd.read_csv(DATA_DIRECTORY, index_col=[0])
    for column in loaded_data.columns:
        if column not in raw_data_schema:
            flag = 1
    if flag == 0:
        print('Raw datas schema is in line with the schema present in schema.py')
    else:
        print('Raw datas schema is NOT in line with the schema present in schema.py')
        
   
###############################################################################
# Define function to validate model's input schema
############################################################################### 

def model_input_schema_check():
    '''
    This function check if all the columns mentioned in model_input_schema in 
    schema.py are present in table named in 'model_input' in db file.

   
    INPUTS
        DB_FILE_NAME : Name of the database file
        DB_PATH : path where the db file should be present
        model_input_schema : schema of models input data in the form oa list/tuple
                          present as in 'schema.py'

    OUTPUT
        If the schema is in line then prints 
        'Models input schema is in line with the schema present in schema.py'
        else prints
        'Models input schema is NOT in line with the schema present in schema.py'
    
    SAMPLE USAGE
        raw_data_schema_check
    '''
    cnx = sqlite3.connect(DB_PATH+DB_FILE_NAME)
    model_input = pd.read_sql('select * from model_input', cnx)
    flag = 0
    for column in model_input.columns:
        if column not in model_input_schema:
            flag = 1
    if flag == 0:
        print('Models input schema is in line with the schema present in schema.py')
    else:
        print('Models input schema is NOT in line with the schema present in schema.py')
        
   

    
    
