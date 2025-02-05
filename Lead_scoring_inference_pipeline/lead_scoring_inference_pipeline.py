##############################################################################
# Import necessary modules
# #############################################################################

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import importlib.util

def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

utils = module_from_file("utils", "/home/airflow/dags/Lead_scoring_inference_pipeline/utils.py")


###############################################################################
# Define default arguments and create an instance of DAG
# ##############################################################################

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2022,7,30),
    'retries' : 1, 
    'retry_delay' : timedelta(seconds=5)
}


Lead_scoring_inference_dag = DAG(
                dag_id = 'Lead_scoring_inference_pipeline',
                default_args = default_args,
                description = 'Inference pipeline of Lead Scoring system',
                schedule_interval = '@hourly',
                catchup = False
)

###############################################################################
# Create a task for encode_data_task() function with task_id 'encoding_categorical_variables'
# ##############################################################################
op_encode_features = PythonOperator(task_id='encoding_categorical_variables',
                                         python_callable=utils.encode_features,
                                         dag=Lead_scoring_inference_dag)


###############################################################################
# Create a task for load_model() function with task_id 'generating_models_prediction'
# ##############################################################################
op_get_models_prediction = PythonOperator(task_id='generating_models_prediction',
                                         python_callable=utils.get_models_prediction,
                                         dag=Lead_scoring_inference_dag)


###############################################################################
# Create a task for prediction_col_check() function with task_id 'checking_model_prediction_ratio'
# ##############################################################################
op_prediction_ratio_check = PythonOperator(task_id='checking_model_prediction_ratio',
                                         python_callable=utils.prediction_ratio_check,
                                         dag=Lead_scoring_inference_dag)


###############################################################################
# Create a task for input_features_check() function with task_id 'checking_input_features'
# ##############################################################################
op_input_features_check = PythonOperator(task_id='checking_input_features',
                                         python_callable=utils.input_features_check,
                                         dag=Lead_scoring_inference_dag)


###############################################################################
# Define relation between tasks
# ##############################################################################

op_encode_features.set_downstream(op_input_features_check)
op_input_features_check.set_downstream(op_get_models_prediction)
op_get_models_prediction.set_downstream(op_prediction_ratio_check)