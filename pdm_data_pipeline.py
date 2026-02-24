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
Executor pods auto-mount user PVC at /mnt/user, so project code and data
are accessible at /mnt/user/pdm-demo/ without custom images.
Spark jobs are submitted via SparkKubernetesOperator to the AIE Spark cluster.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.providers.cncf.kubernetes.operators.spark_kubernetes import SparkKubernetesOperator
from airflow.providers.cncf.kubernetes.sensors.spark_kubernetes import SparkKubernetesSensor
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule


# ---------------------------------------------------------------------------
# DAG configuration
# ---------------------------------------------------------------------------

NAMESPACE = "pdm-demo"
SPARK_IMAGE = "caovd/pdm-spark-etl:latest"
SERVICE_ACCOUNT = "spark"

# Paths on the user PVC (auto-mounted in every executor pod at /mnt/user)
PROJECT_ROOT = "/mnt/user/pdm-demo"
STORAGE_ROOT = "/mnt/user/pdm-demo/data"

# Spark jobs use PVC pv-datasets mounted at /mnt/data inside the Spark container
SPARK_STORAGE_ROOT = "/mnt/data"
SPARK_PROJECT_ROOT = "/opt/pdm"
DATA_PVC = "pv-datasets"

# Pip install command to ensure dependencies are available in executor pods
PIP_INSTALL = "pip install --quiet --break-system-packages pandas requests pyarrow 2>/dev/null; "

default_args = {
    "owner": "Daniel Cao",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}


# ---------------------------------------------------------------------------
# Helper: build bash command with correct env and working dir
# ---------------------------------------------------------------------------

def _bash_cmd(python_module: str, *args: str) -> str:
    """Build a bash command that sets up env, installs deps, and runs a script."""
    cmd_args = " ".join(args) if args else ""
    return (
        f"{PIP_INSTALL}"
        f"export PYTHONPATH={PROJECT_ROOT}:$PYTHONPATH && "
        f"export PDM_STORAGE_ROOT={STORAGE_ROOT} && "
        f"cd {PROJECT_ROOT} && "
        f"python -m {python_module} {cmd_args}"
    )


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
            "mainApplicationFile": f"local://{SPARK_PROJECT_ROOT}/{main_file}",
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
                    {"name": "data-volume", "mountPath": SPARK_STORAGE_ROOT}
                ],
                "env": [
                    {"name": "PDM_STORAGE_ROOT", "value": SPARK_STORAGE_ROOT},
                ],
            },
            "executor": {
                "cores": executor_cores,
                "instances": num_executors,
                "memory": executor_memory,
                "volumeMounts": [
                    {"name": "data-volume", "mountPath": SPARK_STORAGE_ROOT}
                ],
                "env": [
                    {"name": "PDM_STORAGE_ROOT", "value": SPARK_STORAGE_ROOT},
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

    ingest_cmapss = BashOperator(
        task_id="ingest_cmapss",
        bash_command=_bash_cmd(
            "scripts.ingestion.ingest_cmapss",
            "--source", "nasa", "--subsets", "FD001",
        ),
        doc_md="Download and stage NASA C-MAPSS FD001 dataset.",
    )

    ingest_maintnet = BashOperator(
        task_id="ingest_maintnet",
        bash_command=_bash_cmd(
            "scripts.ingestion.ingest_maintnet",
            "--domains", "aviation",
        ),
        doc_md="Download and stage MaintNet aviation logbook datasets + language resources.",
    )

    # =======================================================================
    # Validation Gate: Raw data
    # =======================================================================

    validate_raw_cmapss = BashOperator(
        task_id="validate_raw_cmapss",
        bash_command=_bash_cmd(
            "scripts.validation.validate",
            "--stage", "raw", "--dataset", "cmapss", "--subset", "FD001",
            "--output-dir", f"{STORAGE_ROOT}/validation_reports/",
        ),
        doc_md="Validate raw C-MAPSS data: row counts, schema, nulls, monotonic cycles.",
    )

    validate_raw_maintnet = BashOperator(
        task_id="validate_raw_maintnet",
        bash_command=_bash_cmd(
            "scripts.validation.validate",
            "--stage", "raw", "--dataset", "maintnet", "--domain", "aviation",
            "--output-dir", f"{STORAGE_ROOT}/validation_reports/",
        ),
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

    validate_sensor_processed = BashOperator(
        task_id="validate_sensor_processed",
        bash_command=_bash_cmd(
            "scripts.validation.validate",
            "--stage", "processed", "--dataset", "cmapss", "--subset", "FD001",
            "--output-dir", f"{STORAGE_ROOT}/validation_reports/",
        ),
    )

    validate_sensor_features = BashOperator(
        task_id="validate_sensor_features",
        bash_command=_bash_cmd(
            "scripts.validation.validate",
            "--stage", "features", "--dataset", "cmapss", "--subset", "FD001",
            "--output-dir", f"{STORAGE_ROOT}/validation_reports/",
        ),
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

    validate_text_processed = BashOperator(
        task_id="validate_text_processed",
        bash_command=_bash_cmd(
            "scripts.validation.validate",
            "--stage", "processed", "--dataset", "maintnet", "--domain", "aviation",
            "--output-dir", f"{STORAGE_ROOT}/validation_reports/",
        ),
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
        doc_md="Log completion message.",
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
