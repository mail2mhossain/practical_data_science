import os
import datetime

from typing import Text

from tfx.orchestration import metadata, pipeline
from tfx.orchestration.airflow.airflow_dag_runner import AirflowDagRunner
from tfx.orchestration.airflow.airflow_dag_runner import AirflowPipelineConfig
from base_pipeline import init_components


pipeline_name = "Census_Income_pipeline_airflow"
base_dir = os.getcwd()
data_dir = os.path.join(base_dir, "data")
airflow_dir = os.path.join(base_dir, "airflow")
transform_file = os.path.join(base_dir, "transform.py")
tuner_file = os.path.join(base_dir, "tuner.py")
train_file = os.path.join(base_dir, "trainer.py")

pipeline_root = os.path.join(base_dir, "tfx")
metadata_path = os.path.join(pipeline_root, "metadata.sqlite")
serving_model_dir = os.path.join(pipeline_root, "serving_model")

airflow_config = {
    "schedule_interval": None,
    "start_date": datetime.datetime(2020, 8, 3),
}


def init_pipeline(
    components, pipeline_root: Text, direct_num_workers: int
) -> pipeline.Pipeline:

    beam_arg = [
        f"--direct_num_workers={direct_num_workers}",
    ]
    p = pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
        beam_pipeline_args=beam_arg,
    )
    return p


components = init_components(
    data_dir,
    transform_file,
    tuner_file,
    train_file,
    serving_model_dir
)
pipeline = init_pipeline(components, pipeline_root, 0)
DAG = AirflowDagRunner(AirflowPipelineConfig(airflow_config)).run(pipeline)
