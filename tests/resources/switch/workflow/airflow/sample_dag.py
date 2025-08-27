"""
Sample Airflow DAG for E2E testing
This is a minimal test file to verify Switch conversion functionality
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator

# Default arguments for the DAG
default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

# Define the DAG
dag = DAG(
    'sample_etl_pipeline',
    default_args=default_args,
    description='Sample ETL pipeline for testing',
    schedule_interval=timedelta(days=1),
    catchup=False
)

def extract_data(**context):
    """Extract data from source"""
    print("Extracting data from source system...")
    return "extracted_data"

def transform_data(**context):
    """Transform extracted data"""
    print("Transforming data...")
    return "transformed_data"

def load_data(**context):
    """Load data to destination"""
    print("Loading data to destination...")
    return "data_loaded"

# Define tasks
extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag
)

transform_task = PythonOperator(
    task_id='transform_data',
    python_callable=transform_data,
    dag=dag
)

load_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag
)

validation_task = BashOperator(
    task_id='validate_data',
    bash_command='echo "Data validation completed successfully"',
    dag=dag
)

# Define task dependencies
extract_task >> transform_task >> load_task >> validation_task