##############################################################################
# Import the necessary modules
# #############################################################################

import pandas as pd
import os
import sqlite3
from sqlite3 import Error
from constants import *
import importlib.util
import pytest

def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

utils = module_from_file("utils", "/home/Assignment/01_data_pipeline/scripts/unit_test/utils.py")

###############################################################################
# Write test cases for load_data_into_db() function
# ##############################################################################

def test_load_data_into_db():
    """_summary_
    This function checks if the load_data_into_db function is working properly by
    comparing its output with test cases provided in the db in a table named
    'loaded_data_test_case'

    INPUTS
        DB_FILE_NAME : Name of the database file 'utils_output.db'
        DB_PATH : path where the db file should be present
        UNIT_TEST_DB_FILE_NAME: Name of the test database file 'unit_test_cases.db'

    SAMPLE USAGE
        output=test_get_data()

    """
    cnx = sqlite3.connect(UNIT_TEST_DB_FILE_NAME)
    df = pd.read_sql('select * from loaded_data_test_case', cnx)
    
    cnx1 = sqlite3.connect(DB_PATH + DB_FILE_NAME)
    utils.load_data_into_db()
    df1 = pd.read_sql('select * from loaded_data', cnx1)
    
    assert df.equals(df1), 'the dataframes are not equal'
    assert len(df) == len(df1), 'length of dataframes are not matching'
    
    

###############################################################################
# Write test cases for map_city_tier() function
# ##############################################################################
def test_map_city_tier():
    """_summary_
    This function checks if map_city_tier function is working properly by
    comparing its output with test cases provided in the db in a table named
    'city_tier_mapped_test_case'

    INPUTS
        DB_FILE_NAME : Name of the database file 'utils_output.db'
        DB_PATH : path where the db file should be present
        UNIT_TEST_DB_FILE_NAME: Name of the test database file 'unit_test_cases.db'

    SAMPLE USAGE
        output=test_map_city_tier()

    """
    cnx = sqlite3.connect(UNIT_TEST_DB_FILE_NAME)
    df = pd.read_sql('select * from city_tier_mapped_test_case', cnx)
    print(df.columns)
    cnx1 = sqlite3.connect(DB_PATH + DB_FILE_NAME)
    utils.map_city_tier()
    df1 = pd.read_sql('select * from city_teir_mapped', cnx1)
    print(df1.columns)
    
    assert df.equals(df1), 'the dataframes are not equal'
    assert len(df) == len(df1), 'length of dataframes are not matching'
    
###############################################################################
# Write test cases for map_categorical_vars() function
# ##############################################################################    
def test_map_categorical_vars():
    """_summary_
    This function checks if map_cat_vars function is working properly by
    comparing its output with test cases provided in the db in a table named
    'categorical_variables_mapped_test_case'

    INPUTS
        DB_FILE_NAME : Name of the database file 'utils_output.db'
        DB_PATH : path where the db file should be present
        UNIT_TEST_DB_FILE_NAME: Name of the test database file 'unit_test_cases.db'
    
    SAMPLE USAGE
        output=test_map_cat_vars()

    """    
    cnx = sqlite3.connect(UNIT_TEST_DB_FILE_NAME)
    df = pd.read_sql('select * from categorical_variables_mapped_test_case', cnx)
    print(df.columns)
    cnx1 = sqlite3.connect(DB_PATH + DB_FILE_NAME)
    utils.map_categorical_vars()
    df1 = pd.read_sql('select * from categorical_variables_mapped', cnx1)
    print(df1.columns)
      
    assert df.equals(df1), 'the dataframes are not equal'
    assert len(df) == len(df1), 'length of dataframes are not matching'

###############################################################################
# Write test cases for interactions_mapping() function
# ##############################################################################    
def test_interactions_mapping():
    """_summary_
    This function checks if test_column_mapping function is working properly by
    comparing its output with test cases provided in the db in a table named
    'interactions_mapped_test_case'

    INPUTS
        DB_FILE_NAME : Name of the database file 'utils_output.db'
        DB_PATH : path where the db file should be present
        UNIT_TEST_DB_FILE_NAME: Name of the test database file 'unit_test_cases.db'

    SAMPLE USAGE
        output=test_column_mapping()

    """ 
    cnx = sqlite3.connect(UNIT_TEST_DB_FILE_NAME)
    df = pd.read_sql('select * from interactions_mapped_test_case', cnx)
    print(df.columns)
    cnx1 = sqlite3.connect(DB_PATH + DB_FILE_NAME)
    utils.interactions_mapping()
    df1 = pd.read_sql('select * from interactions_mapped', cnx1)
    print(df1.columns)
     
    assert df.equals(df1), 'the dataframes are not equal'
    assert len(df) == len(df1), 'length of dataframes are not matching'