from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.dummy import DummyOperator
from docker.types import Mount



with DAG(
    'etl_pipeline2',
    description='Pipeline ETL: Normalizer -> Snowflake',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:

    start = DummyOperator(task_id='start')

    run_normalizer = DockerOperator(
        task_id='run_normalizer',
        image='ilanlp/normalizer:latest',
        api_version='auto',
        auto_remove=True,
        network_mode='jm_network',
        mounts=[
            Mount(source='/home/ilanlp/Projet_backend/pipeline/src/.env', target='/app/.env', type='bind', read_only=True),
            Mount(source='/home/ilanlp/Projet_backend/pipeline/src/data', target='/usr/src/data', type='bind', read_only=False),
        ],
    )

    run_snowflake_init = DockerOperator(
        task_id='run_snowflake_init',
        image='ilanlp/snowflake_init:latest',
        api_version='auto',
        auto_remove=True,
        network_mode='jm_network',
        mounts=[
            Mount(source='/home/ilanlp/Projet_backend/pipeline/src/data', target='/usr/src/data', type='bind', read_only=False),
            
        ],
        environment={
            "SNOWFLAKE_USER": "ILANLP",
            "SNOWFLAKE_PASSWORD": "Cll3k#lj1o^5Wq",
            "SNOWFLAKE_ACCOUNT": "veb24721.us-west-2",
            "SNOWFLAKE_WAREHOUSE": "QUERY",
            "SNOWFLAKE_DATABASE": "PROJECT_DBT",
            "SNOWFLAKE_SCHEMA": "PUBLIC",
            "SNOWFLAKE_ROLE": "ACCOUNTADMIN",
        },
    )

    run_snowflake_core = DockerOperator(
        task_id='run_snowflake_core',
        image='ilanlp/snowflake_core:latest',
        api_version='auto',
        auto_remove=True,
        network_mode='jm_network',
        mounts=[
            Mount(source='/home/ilanlp/Projet_backend/pipeline/src/data', target='/usr/src/data', type='bind', read_only=False),
            
        ],
        environment={
            "SNOWFLAKE_USER": "ILANLP",
            "SNOWFLAKE_PASSWORD": "Cll3k#lj1o^5Wq",
            "SNOWFLAKE_ACCOUNT": "veb24721.us-west-2",
            "SNOWFLAKE_WAREHOUSE": "QUERY",
            "SNOWFLAKE_DATABASE": "PROJECT_DBT",
            "SNOWFLAKE_SCHEMA": "PUBLIC",
            "SNOWFLAKE_ROLE": "ACCOUNTADMIN",
        },
    )

    run_snowflake_alljob = DockerOperator(
        task_id='run_snowflake_alljob',
        image='ilanlp/alljob:latest',
        api_version='auto',
        auto_remove=True,
        network_mode='jm_network',
        mounts=[
            Mount(source='/home/ilanlp/Projet_backend/pipeline/src/data', target='/usr/src/data', type='bind', read_only=False),
            
        ],
        environment={
            "SNOWFLAKE_USER": "ILANLP",
            "SNOWFLAKE_PASSWORD": "Cll3k#lj1o^5Wq",
            "SNOWFLAKE_ACCOUNT": "veb24721.us-west-2",
            "SNOWFLAKE_WAREHOUSE": "QUERY",
            "SNOWFLAKE_DATABASE": "PROJECT_DBT",
            "SNOWFLAKE_SCHEMA": "PUBLIC",
            "SNOWFLAKE_ROLE": "ACCOUNTADMIN",
        },
    )

    run_dbt = DockerOperator(
        task_id='run_dbt',
        image='jm-elt-dbt:latest',
        container_name='airflow_jm_elt_dbt',
        api_version='auto',
        auto_remove=True,
        docker_url='unix://var/run/docker.sock',
        network_mode='jm_network',
        mounts=[
            Mount(source='/home/ilanlp/Projet_backend/snowflake/DBT', target='/usr/src/DBT', type='bind', read_only=False),
            Mount(source='/home/ilanlp/Projet_backend/snowflake/DBT/.env', target='/usr/src/DBT/.env', type='bind', read_only=True),
        ],
    )

    end = DummyOperator(task_id='end')

    start >> run_normalizer >> run_snowflake_init >> run_snowflake_core >> run_snowflake_alljob >> run_dbt >> end