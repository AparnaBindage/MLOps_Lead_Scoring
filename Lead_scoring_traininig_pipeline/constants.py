rootfolder = "/home/"
DB_PATH = rootfolder + "airflow/dags/Lead_scoring_data_pipeline/"
DB_FILE_NAME = "lead_scoring_data_cleaning.db"

DB_FILE_MLFLOW = rootfolder + "airflow/dags/Lead_scoring_training_pipeline/Lead_scoring_mlflow_production.db"

TRACKING_URI = "http://0.0.0.0:6006"
EXPERIMENT = 'Lead_scoring_mlflow_production'


# model config imported from pycaret experimentation
model_config =  {'bagging_fraction' : 0.6, 
        'bagging_freq': 3, 
        'boosting_type' : 'gbdt',
        'class_weight' : None, 
        'colsample_bytree' : 1.0,
        'feature_fraction' : 0.7,
        'importance_type' : 'split',
        'learning_rate' : 0.2,
        'max_depth' : -1,
        'min_child_samples' : 81,
        'min_child_weight' : 0.001,
        'min_split_gain' : 0.1,
        'n_estimators' : 90,
        'n_jobs' : -1, 
        'num_leaves' : 40,
        'objective' : None,
        'random_state' : 42, 
        'reg_alpha' : 0.0005,
        'reg_lambda' : 0.7, 
        'silent' : 'warn',
        'subsample' : 1.0, 
        'subsample_for_bin' : 200000,
        'subsample_freq' : 0}

# list of the features that needs to be there in the final encoded dataframe
ONE_HOT_ENCODED_FEATURES = ['total_leads_droppped', 'city_tier', 'referred_lead', 'app_complete_flag', 'first_platform_c', 'first_utm_medium_c', 'first_utm_source_c']
# list of features that need to be one-hot encoded
FEATURES_TO_ENCODE = ['first_platform_c','first_utm_medium_c','first_utm_source_c']
