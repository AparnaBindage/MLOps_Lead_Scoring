##############################################################################
# Import necessary modules and files
# #############################################################################


import pandas as pd
import os
import sqlite3
from sqlite3 import Error
from constants import *
from city_tier_mapping import *
from significant_categorical_level import *
###############################################################################
# Define the function to build database
# ##############################################################################

def build_dbs():
    '''
    This function checks if the db file with specified name is present 
    in the /Assignment/01_data_pipeline/scripts folder. If it is not present it creates 
    the db file with the given name at the given path. 


    INPUTS
        DB_FILE_NAME : Name of the database file 'utils_output.db'
        DB_PATH : path where the db file should exist  


    OUTPUT
    The function returns the following under the given conditions:
        1. If the file exists at the specified path
                prints 'DB Already Exists' and returns 'DB Exists'

        2. If the db file is not present at the specified loction
                prints 'Creating Database' and creates the sqlite db 
                file at the specified path with the specified name and 
                once the db file is created prints 'New DB Created' and 
                returns 'DB created'


    SAMPLE USAGE
        build_dbs()
    '''
    if os.path.isfile(DB_PATH+DB_FILE_NAME):
        print( "DB Already Exists")
        print(os.getcwd())
        return "DB Exists"
    else:
        print ("Creating Database")
        """ create a database connection to a SQLite database """
        conn = None
        try: 
            conn = sqlite3.connect(DB_PATH+DB_FILE_NAME)
            print("New DB Created")
        except Error as e:
            print(e)
            return "Error"
        finally:
            if conn:
                conn.close()
                return "DB Created"

def check_if_table_has_value(cnx,table_name):
    cnx = sqlite3.connect(DB_PATH+DB_FILE_NAME)
    check_table = pd.read_sql(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';", cnx).shape[0]
    if check_table == 1:
        return True
    else:
        return False



###############################################################################
# Define function to load the csv file to the database
# ##############################################################################

def load_data_into_db():
    '''
    Thie function loads the data present in data directory into the db
    which was created previously.
    It also replaces any null values present in 'toal_leads_dropped' and
    'referred_lead' columns with 0.


    INPUTS
        DB_FILE_NAME : Name of the database file
        DB_PATH : path where the db file should be
        DATA_DIRECTORY : path of the directory where 'leadscoring.csv' 
                        file is present
        

    OUTPUT
        Saves the processed dataframe in the db in a table named 'loaded_data'.
        If the table with the same name already exsists then the function 
        replaces it.


    SAMPLE USAGE
        load_data_into_db()
    '''
    cnx = sqlite3.connect(DB_PATH+DB_FILE_NAME)

    if not check_if_table_has_value(cnx,'loaded_data'):
        print("Table Doesn't Exsist - loaded_data, Building")
     
        loaded_data = pd.read_csv(DATA_DIRECTORY)
        print(loaded_data["total_leads_droppped"].isna().sum())
        print(loaded_data["referred_lead"].isna().sum())
        loaded_data["total_leads_droppped"] = get_fill_na_dataframe(loaded_data, 'total_leads_droppped', value=0)
        loaded_data["referred_lead"] = get_fill_na_dataframe(loaded_data, 'referred_lead', value=0)
        print(loaded_data["total_leads_droppped"].isna().sum())
        print(loaded_data["referred_lead"].isna().sum())
        print(loaded_data.shape)
        print(loaded_data.columns)
        loaded_data.to_sql(name='loaded_data', con=cnx,if_exists='replace',index=False)
        


###############################################################################
# Define function to map cities to their respective tiers
# ##############################################################################

def get_label_encoding_dataframe(dataframe, column_name, mapping_dict):
    return dataframe[column_name].map(mapping_dict) 

def get_fill_na_dataframe(dataframe, column_name, value='mean'):
    if value != 'mean' and value !='mode':
        return dataframe[column_name].fillna(value)
    elif value == 'mean':
        value = dataframe[column_name].mean()
        return dataframe[column_name].fillna(value)
    elif value == 'mode':
        value = dataframe[column_name].mode()
        return dataframe[column_name].fillna(value)
    
def map_city_tier():
    '''
    This function maps all the cities to their respective tier as per the
    mappings provided in the city_tier_mapping.py file. If a
    particular city's tier isn't mapped(present) in the city_tier_mapping.py 
    file then the function maps that particular city to 3.0 which represents
    tier-3.


    INPUTS
        DB_FILE_NAME : Name of the database file
        DB_PATH : path where the db file should be
        city_tier_mapping : a dictionary that maps the cities to their tier

    
    OUTPUT
        Saves the processed dataframe in the db in a table named
        'city_tier_mapped'. If the table with the same name already 
        exsists then the function replaces it.

    
    SAMPLE USAGE
        map_city_tier()

    '''
    cnx = sqlite3.connect(DB_PATH+DB_FILE_NAME)
    df_lead_scoring = pd.read_sql('select * from loaded_data', cnx)
    #print(df_lead_scoring.shape)
    
    if not check_if_table_has_value(cnx,'city_teir_mapped'):
        df_lead_scoring['city_tier'] = get_label_encoding_dataframe(df_lead_scoring, 'city_mapped',city_tier_mapping)
        df_lead_scoring["city_tier"] = get_fill_na_dataframe(df_lead_scoring, 'city_tier', value=3.0)
        #print(df_lead_scoring.shape)
        df_lead_scoring = df_lead_scoring.drop(['city_mapped'], axis = 1)
        df_lead_scoring.to_sql(name='city_teir_mapped', con=cnx,if_exists='replace',index=False)

###############################################################################
# Define function to map insignificant categorial variables to "others"
# ##############################################################################


def map_categorical_vars():
    '''
    This function maps all the insignificant variables present in 'first_platform_c'
    'first_utm_medium_c' and 'first_utm_source_c'. The list of significant variables
    should be stored in a python file in the 'significant_categorical_level.py' 
    so that it can be imported as a variable in utils file.
    

    INPUTS
        DB_FILE_NAME : Name of the database file
        DB_PATH : path where the db file should be present
        list_platform : list of all the significant platform.
        list_medium : list of all the significat medium
        list_source : list of all the significant source

        **NOTE : list_platform, list_medium & list_source are all constants and
                 must be stored in 'significant_categorical_level.py'
                 file. The significant levels are calculated by taking top 90
                 percentils of all the levels. For more information refer
                 'data_cleaning.ipynb' notebook.
  

    OUTPUT
        Saves the processed dataframe in the db in a table named
        'categorical_variables_mapped'. If the table with the same name already 
        exsists then the function replaces it.

    
    SAMPLE USAGE
        map_categorical_vars()
    '''
    cnx = sqlite3.connect(DB_PATH+DB_FILE_NAME)
    df_lead_scoring = pd.read_sql('select * from city_teir_mapped', cnx)

    if not check_if_table_has_value(cnx,'categorical_variables_mapped'):
        new_df = df_lead_scoring[~df_lead_scoring['first_platform_c'].isin(list_platform)] # get rows for levels which are not present in list_platform
        new_df['first_platform_c'] = "others" # replace the value of these levels to others
        old_df = df_lead_scoring[df_lead_scoring['first_platform_c'].isin(list_platform)] # get rows for levels which are present in list_platform
        df = pd.concat([new_df, old_df]) # concatenate new_df and old_df to get the final dataframe
        
        new_df = df[~df['first_utm_medium_c'].isin(list_medium)] # get rows for levels which are not present in list_medium
        new_df['first_utm_medium_c'] = "others" # replace the value of these levels to others
        old_df = df[df['first_utm_medium_c'].isin(list_medium)] # get rows for levels which are present in list_medium
        df = pd.concat([new_df, old_df]) # concatenate new_df and old_df to get the final dataframe

        # all the levels below 90 percentage are assgined to a single level called others
        new_df = df[~df['first_utm_source_c'].isin(list_source)] # get rows for levels which are not present in list_source
        new_df['first_utm_source_c'] = "others" # replace the value of these levels to others
        old_df = df[df['first_utm_source_c'].isin(list_source)] # get rows for levels which are present in list_source
        df = pd.concat([new_df, old_df]) # concatenate new_df and old_df to get the final dataframe
        
        df["total_leads_droppped"] = get_fill_na_dataframe(df, 'total_leads_droppped', value=0)
        df['referred_lead'] = get_fill_na_dataframe(df, 'referred_lead', value=0)
        
        print(df.shape)
        df.to_sql(name='categorical_variables_mapped', con=cnx,if_exists='replace',index=False)

##############################################################################
# Define function that maps interaction columns into 4 types of interactions
# #############################################################################
def interactions_mapping():
    '''
    This function maps the interaction columns into 4 unique interaction columns
    These mappings are present in 'interaction_mapping.csv' file. 


    INPUTS
        DB_FILE_NAME: Name of the database file
        DB_PATH : path where the db file should be present
        INTERACTION_MAPPING : path to the csv file containing interaction's
                                   mappings
        INDEX_COLUMNS_TRAINING : list of columns to be used as index while pivoting and
                                 unpivoting during training
        INDEX_COLUMNS_INFERENCE: list of columns to be used as index while pivoting and
                                 unpivoting during inference
        NOT_FEATURES: Features which have less significance and needs to be dropped
                                 
        NOTE : Since while inference we will not have 'app_complete_flag' which is
        our label, we will have to exculde it from our features list. It is recommended 
        that you use an if loop and check if 'app_complete_flag' is present in 
        'categorical_variables_mapped' table and if it is present pass a list with 
        'app_complete_flag' column, or else pass a list without 'app_complete_flag'
        column.

    
    OUTPUT
        Saves the processed dataframe in the db in a table named 
        'interactions_mapped'. If the table with the same name already exsists then 
        the function replaces it.
        
        It also drops all the features that are not requried for training model and 
        writes it in a table named 'model_input'

    
    SAMPLE USAGE
        interactions_mapping()
    '''
    cnx = sqlite3.connect(DB_PATH+DB_FILE_NAME)
    df = pd.read_sql('select * from categorical_variables_mapped', cnx)

    if not check_if_table_has_value(cnx,'interactions_mapped'):
        df = df.drop_duplicates()
 
        # unpivot the interaction columns and put the values in rows
        df_unpivot = pd.melt(df, id_vars=INDEX_COLUMNS_TRAINING, var_name='interaction_type', value_name='interaction_value')
        # 37 interaction columns, each row gets converted into 37 rows 

        
        # handle the nulls in the interaction value column
        df_unpivot['interaction_value'] = get_fill_na_dataframe(df_unpivot, 'interaction_value', value=0)

        # read the interaction mapping file
        df_event_mapping = pd.read_csv(INTERACTION_MAPPING, index_col=[0])
   
        # map interaction type column with the mapping file to get interaction mapping
        df = pd.merge(df_unpivot, df_event_mapping, on='interaction_type', how='left')
       
        
        #dropping the interaction type column as it is not needed
        df = df.drop(['interaction_type'], axis=1)
       
        # pivoting the interaction mapping column values to individual columns in the dataset
        df_pivot = df.pivot_table(
                values='interaction_value', index=INDEX_COLUMNS_TRAINING, columns='interaction_mapping', aggfunc='sum')
        df_pivot = df_pivot.reset_index()
        # the rows are again converted back to columns. 37 rows gets converted to a single row.
        
        
        # the file is written in the data folder
        #df_pivot.to_csv('Data/cleaned_data.csv', index=False)
        df_pivot.to_sql(name='interactions_mapped', con=cnx,if_exists='replace',index=False)
        df_pivot.drop(NOT_FEATURES,axis=1,inplace=True)
        df_pivot.to_sql(name='model_input', con=cnx,if_exists='replace',index=False)
        
