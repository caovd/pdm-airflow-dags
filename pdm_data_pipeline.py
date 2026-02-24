"""
Airflow DAG: pdm_data_pipeline

Orchestrates the end-to-end data ingestion and ETL pipeline for the
Predictive Maintenance demo on HPE Private Cloud AI.

Pipeline flow:
  [ingest_cmapss] ──┐
                    ├──[validate_raw]──[fork]──┬──[sensor_etl_spark]──[validate_sensor_processed]──┐
  [ingest_maintnet]─┘                          │                                                    ├──[notify_complete]
                                               └──[text_etl_spark]────[validate_text_processed]────┘

Runs on HPE AI Essentials' built-in Apache Airflow.
Spark jobs are submitted via SparkKubernetesOperator to the AIE Spark cluster.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.providers.cncf.kubernetes.operators.spark_kubernetes import SparkKubernetesOperator
from airflow.providers.cncf.kubernetes.sensors.spark_kubernetes import SparkKubernetesSensor
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule
from kubernetes.client import models as k8s


# ---------------------------------------------------------------------------
# DAG configuration
# ---------------------------------------------------------------------------

NAMESPACE = "pdm-demo"
USER_NAMESPACE = "project-user-daniel-cao"
SPARK_IMAGE = "caovd/pdm-spark-etl:latest"
SERVICE_ACCOUNT = "spark"
DATA_PVC = "pv-datasets"
STORAGE_ROOT = "/mnt/data"
PROJECT_ROOT = "/opt/pdm"

default_args = {
    "owner": "Daniel Cao",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}

# ---------------------------------------------------------------------------
# Shared volume config for KubernetesPodOperator tasks
# ---------------------------------------------------------------------------

DATA_VOLUME = k8s.V1Volume(
    name="data-volume",
    persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
        claim_name=DATA_PVC,
    ),
)

DATA_VOLUME_MOUNT = k8s.V1VolumeMount(
    name="data-volume",
    mount_path=STORAGE_ROOT,
)

POD_ENV_VARS = {
    "PDM_STORAGE_ROOT": STORAGE_ROOT,
    "PYTHONPATH": PROJECT_ROOT,
}


# ---------------------------------------------------------------------------
# Helper: SparkApplication YAML generator
# ---------------------------------------------------------------------------

def spark_application_yaml(
    name: str,
    main_file: str,
    args: list = None,
    driver_memory: str = "4g",
    executor_memory: str = "8g",
    executor_cores: int = 4,
    num_executors: int = 2,
    gpu: int = 0,
) -> dict:
    """Generate a SparkApplication spec for SparkKubernetesOperator."""
    spec = {
        "apiVersion": "sparkoperator.k8s.io/v1beta2",
        "kind": "SparkApplication",
        "metadata": {
            "name": name,
            "namespace": NAMESPACE,
        },
        "spec": {
            "type": "Python",
            "mode": "cluster",
            "image": SPARK_IMAGE,
            "imagePullPolicy": "Always",
            "mainApplicationFile": f"local://{PROJECT_ROOT}/{main_file}",
            "arguments": args or [],
            "sparkVersion": "3.5.0",
            "restartPolicy": {"type": "OnFailure", "onFailureRetries": 2},
            "volumes": [
                {
                    "name": "data-volume",
                    "persistentVolumeClaim": {"claimName": DATA_PVC},
                }
            ],
            "driver": {
                "cores": 1,
                "memory": driver_memory,
                "serviceAccount": SERVICE_ACCOUNT,
                "volumeMounts": [
                    {"name": "data-volume", "mountPath": STORAGE_ROOT}
                ],
                "env": [
                    {"name": "PDM_STORAGE_ROOT", "value": STORAGE_ROOT},
                ],
            },
            "executor": {
                "cores": executor_cores,
                "instances": num_executors,
                "memory": executor_memory,
                "volumeMounts": [
                    {"name": "data-volume", "mountPath": STORAGE_ROOT}
                ],
                "env": [
                    {"name": "PDM_STORAGE_ROOT", "value": STORAGE_ROOT},
                ],
            },
        },
    }

    if gpu > 0:
        spec["spec"]["executor"]["gpu"] = {"name": "nvidia.com/gpu", "quantity": gpu}

    return spec


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="pdm_data_pipeline",
    default_args=default_args,
    description="Predictive Maintenance — Data Ingestion & ETL Pipeline",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["pdm-demo", "data-engineering", "phase-1-2"],
    doc_md=__doc__,
    max_active_runs=1,
) as dag:

    # =======================================================================
    # Phase 1: Data Ingestion
    # =======================================================================

    start = EmptyOperator(task_id="start")

    ingest_cmapss = KubernetesPodOperator(
        task_id="ingest_cmapss",
        namespace=USER_NAMESPACE,
        image=SPARK_IMAGE,
        cmds=["python", "-m", "scripts.ingestion.ingest_cmapss"],
        arguments=["--source", "nasa", "--subsets", "FD001"],
        env_vars=POD_ENV_VARS,
        working_dir=PROJECT_ROOT,
        volumes=[DATA_VOLUME],
        volume_mounts=[DATA_VOLUME_MOUNT],
        name="ingest-cmapss",
        get_logs=True,
        is_delete_operator_pod=True,
        doc_md="Download and stage NASA C-MAPSS FD001 dataset.",
    )

    ingest_maintnet = KubernetesPodOperator(
        task_id="ingest_maintnet",
        namespace=USER_NAMESPACE,
        image=SPARK_IMAGE,
        cmds=["python", "-m", "scripts.ingestion.ingest_maintnet"],
        arguments=["--domains", "aviation"],
        env_vars=POD_ENV_VARS,
        working_dir=PROJECT_ROOT,
        volumes=[DATA_VOLUME],
        volume_mounts=[DATA_VOLUME_MOUNT],
        name="ingest-maintnet",
        get_logs=True,
        is_delete_operator_pod=True,
        doc_md="Download and stage MaintNet aviation logbook datasets + language resources.",
    )

    # =======================================================================
    # Validation Gate: Raw data
    # =======================================================================

    validate_raw_cmapss = KubernetesPodOperator(
        task_id="validate_raw_cmapss",
        namespace=USER_NAMESPACE,
        image=SPARK_IMAGE,
        cmds=["python", "-m", "scripts.validation.validate"],
        arguments=[
            "--stage", "raw",
            "--dataset", "cmapss",
            "--subset", "FD001",
            "--output-dir", f"{STORAGE_ROOT}/validation_reports/",
        ],
        env_vars=POD_ENV_VARS,
        working_dir=PROJECT_ROOT,
        volumes=[DATA_VOLUME],
        volume_mounts=[DATA_VOLUME_MOUNT],
        name="validate-raw-cmapss",
        get_logs=True,
        is_delete_operator_pod=True,
        doc_md="Validate raw C-MAPSS data: row counts, schema, nulls, monotonic cycles.",
    )

    validate_raw_maintnet = KubernetesPodOperator(
        task_id="validate_raw_maintnet",
        namespace=USER_NAMESPACE,
        image=SPARK_IMAGE,
        cmds=["python", "-m", "scripts.validation.validate"],
        arguments=[
            "--stage", "raw",
            "--dataset", "maintnet",
            "--domain", "aviation",
            "--output-dir", f"{STORAGE_ROOT}/validation_reports/",
        ],
        env_vars=POD_ENV_VARS,
        working_dir=PROJECT_ROOT,
        volumes=[DATA_VOLUME],
        volume_mounts=[DATA_VOLUME_MOUNT],
        name="validate-raw-maintnet",
        get_logs=True,
        is_delete_operator_pod=True,
        doc_md="Validate raw MaintNet data: manifest exists, files accessible.",
    )

    raw_validated = EmptyOperator(
        task_id="raw_data_validated",
        trigger_rule=TriggerRule.ALL_SUCCESS,
        doc_md="Gate: both raw datasets validated successfully.",
    )

    # =======================================================================
    # Phase 2: Spark ETL (parallel branches)
    # =======================================================================

    # --- Branch A: C-MAPSS Sensor ETL ---

    sensor_etl_spec = spark_application_yaml(
        name="cmapss-sensor-etl-fd001",
        main_file="scripts/etl/sensor_etl_spark.py",
        args=["--subset", "FD001"],
        driver_memory="4g",
        executor_memory="8g",
        executor_cores=4,
        num_executors=2,
    )

    sensor_etl = SparkKubernetesOperator(
        task_id="sensor_etl_spark",
        namespace=NAMESPACE,
        application_file=sensor_etl_spec,
        do_xcom_push=True,
        doc_md=(
            "Spark job: C-MAPSS sensor ETL.\n"
            "- Drop constant sensors\n"
            "- Operating-condition min-max normalisation\n"
            "- Piecewise-linear RUL labeling (cap=125)\n"
            "- Sliding window creation (30×14)\n"
            "- Train/val/test split by engine\n"
            "- Output: Parquet on GreenLake File Storage"
        ),
    )

    sensor_etl_sensor = SparkKubernetesSensor(
        task_id="sensor_etl_monitor",
        namespace=NAMESPACE,
        application_name="cmapss-sensor-etl-fd001",
        poke_interval=30,
        timeout=3600,
    )

    validate_sensor_processed = KubernetesPodOperator(
        task_id="validate_sensor_processed",
        namespace=USER_NAMESPACE,
        image=SPARK_IMAGE,
        cmds=["python", "-m", "scripts.validation.validate"],
        arguments=[
            "--stage", "processed",
            "--dataset", "cmapss",
            "--subset", "FD001",
            "--output-dir", f"{STORAGE_ROOT}/validation_reports/",
        ],
        env_vars=POD_ENV_VARS,
        working_dir=PROJECT_ROOT,
        volumes=[DATA_VOLUME],
        volume_mounts=[DATA_VOLUME_MOUNT],
        name="validate-sensor-processed",
        get_logs=True,
        is_delete_operator_pod=True,
    )

    validate_sensor_features = KubernetesPodOperator(
        task_id="validate_sensor_features",
        namespace=USER_NAMESPACE,
        image=SPARK_IMAGE,
        cmds=["python", "-m", "scripts.validation.validate"],
        arguments=[
            "--stage", "features",
            "--dataset", "cmapss",
            "--subset", "FD001",
            "--output-dir", f"{STORAGE_ROOT}/validation_reports/",
        ],
        env_vars=POD_ENV_VARS,
        working_dir=PROJECT_ROOT,
        volumes=[DATA_VOLUME],
        volume_mounts=[DATA_VOLUME_MOUNT],
        name="validate-sensor-features",
        get_logs=True,
        is_delete_operator_pod=True,
    )

    # --- Branch B: MaintNet Text ETL ---

    text_etl_spec = spark_application_yaml(
        name="maintnet-text-etl-aviation",
        main_file="scripts/etl/text_etl_spark.py",
        args=["--domain", "aviation"],
        driver_memory="4g",
        executor_memory="4g",
        executor_cores=2,
        num_executors=2,
    )

    text_etl = SparkKubernetesOperator(
        task_id="text_etl_spark",
        namespace=NAMESPACE,
        application_file=text_etl_spec,
        do_xcom_push=True,
        doc_md=(
            "Spark job: MaintNet text ETL.\n"
            "- Schema detection & column mapping\n"
            "- Abbreviation expansion (MaintNet dictionary)\n"
            "- Spell correction (Levenshtein + term bank)\n"
            "- Rule-based fault classification\n"
            "- LLM instruction-tuning format (JSONL)\n"
            "- Output: Parquet + JSONL on GreenLake File Storage"
        ),
    )

    text_etl_sensor = SparkKubernetesSensor(
        task_id="text_etl_monitor",
        namespace=NAMESPACE,
        application_name="maintnet-text-etl-aviation",
        poke_interval=30,
        timeout=3600,
    )

    validate_text_processed = KubernetesPodOperator(
        task_id="validate_text_processed",
        namespace=USER_NAMESPACE,
        image=SPARK_IMAGE,
        cmds=["python", "-m", "scripts.validation.validate"],
        arguments=[
            "--stage", "processed",
            "--dataset", "maintnet",
            "--domain", "aviation",
            "--output-dir", f"{STORAGE_ROOT}/validation_reports/",
        ],
        env_vars=POD_ENV_VARS,
        working_dir=PROJECT_ROOT,
        volumes=[DATA_VOLUME],
        volume_mounts=[DATA_VOLUME_MOUNT],
        name="validate-text-processed",
        get_logs=True,
        is_delete_operator_pod=True,
    )

    # =======================================================================
    # Join and notify
    # =======================================================================

    all_etl_complete = EmptyOperator(
        task_id="all_etl_complete",
        trigger_rule=TriggerRule.ALL_SUCCESS,
        doc_md="Gate: both ETL branches complete and validated.",
    )

    notify_data_ready = BashOperator(
        task_id="notify_data_ready",
        bash_command=(
            'echo "✓ Phase 1-2 data pipeline complete. '
            'Processed data available at:"; '
            f'echo "  Sensor features: {STORAGE_ROOT}/features/cmapss/FD001/"; '
            f'echo "  Text features:   {STORAGE_ROOT}/features/maintnet/aviation/"; '
            'echo "Ready for Phase 3 (Analytics) and Phase 4 (Model Training)."'
        ),
        doc_md="Log completion message. In production, this could trigger a Slack/email notification.",
    )

    # =======================================================================
    # Task dependencies
    # =======================================================================

    start >> [ingest_cmapss, ingest_maintnet]

    ingest_cmapss >> validate_raw_cmapss
    ingest_maintnet >> validate_raw_maintnet

    [validate_raw_cmapss, validate_raw_maintnet] >> raw_validated

    # Parallel ETL branches
    raw_validated >> sensor_etl >> sensor_etl_sensor >> validate_sensor_processed >> validate_sensor_features
    raw_validated >> text_etl >> text_etl_sensor >> validate_text_processed

    # Join
    [validate_sensor_features, validate_text_processed] >> all_etl_complete >> notify_data_ready
