import tensorflow_model_analysis as tfma
from tfx.components import (
    CsvExampleGen,
    Evaluator,
    ExampleValidator,
    Pusher,
    ResolverNode,
    SchemaGen,
    StatisticsGen,
    Trainer,
    Transform,
    Tuner
)
from tfx.components.base import executor_spec
from tfx.components.trainer.executor import GenericExecutor
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.proto import pusher_pb2, trainer_pb2, example_gen_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.utils.dsl_utils import external_input



def init_components(
    data_dir,
    transform_file,
    tuner_file,
    train_file,
    serving_model_dir=None
):


    output_config = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(splits=[
        example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=6),
        example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=2),
        example_gen_pb2.SplitConfig.Split(name='test', hash_buckets=2)
    ]))

    example_gen = CsvExampleGen(input_base=data_dir, output_config=output_config)

    statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])

    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs["statistics"],
        infer_feature_shape=False,
    )

    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )

    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=transform_file,
    )
    
    tuner = Tuner(
        module_file=tuner_file,
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(splits=['train'], num_steps=500),
        eval_args=trainer_pb2.EvalArgs(splits=['eval'], num_steps=100)
    )

    trainer = Trainer(
        module_file=train_file,
        examples=transform.outputs['transformed_examples'],
        hyperparameters=tuner.outputs['best_hyperparameters'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(splits=['train']),
        eval_args=trainer_pb2.EvalArgs(splits=['eval'])
        )

    model_resolver = ResolverNode(
        instance_name="latest_blessed_model_resolver",
        resolver_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing),
    )

    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key="label")],
        slicing_specs=[
            tfma.SlicingSpec(),
            tfma.SlicingSpec(feature_keys=["race"]),
            tfma.SlicingSpec(feature_keys=["sex"]),
        ],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(class_name="BinaryAccuracy"),
                    tfma.MetricConfig(class_name="ExampleCount"),
                    tfma.MetricConfig(class_name="AUC"),
                    tfma.MetricConfig(class_name="Precision"),
                    tfma.MetricConfig(class_name="Recall"),
                ],
                thresholds={
                    "AUC": tfma.config.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={"value": 0.65}
                        ),
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={"value": 0.01},
                        ),
                    )
                },
            )
        ],
    )

    evaluator = Evaluator(
        examples=example_gen.outputs["examples"],
        model=trainer.outputs["model"],
        baseline_model=model_resolver.outputs["model"],
        eval_config=eval_config,
    )

    pusher_kwargs = {
        "model": trainer.outputs["model"],
        "model_blessing": evaluator.outputs["blessing"],
        "push_destination": pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
            base_directory=serving_model_dir
        )
    )
    }


    pusher = Pusher(**pusher_kwargs)

    components = [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        trainer,
        model_resolver,
        evaluator,
        pusher,
    ]
    return components
