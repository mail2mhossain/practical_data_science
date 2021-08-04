
import os
from tensorflow import keras
from typing import NamedTuple, Dict, Text, Any, List
from tfx.components.trainer.fn_args_utils import FnArgs, DataAccessor
import tensorflow as tf
import tensorflow_transform as tft

from tensorflow_transform.tf_metadata import schema_utils
from tfx_bsl.public.tfxio import TensorFlowDatasetOptions

# import same constants from transform module
import census_constants

# Unpack the contents of the constants module
_NUMERIC_FEATURE_KEYS = census_constants.NUMERIC_FEATURE_KEYS
_VOCAB_FEATURE_DICT = census_constants.VOCAB_FEATURE_DICT
_BUCKET_FEATURE_DICT = census_constants.BUCKET_FEATURE_DICT
_NUM_OOV_BUCKETS = census_constants.NUM_OOV_BUCKETS
_LABEL_KEY = census_constants.LABEL_KEY
_transformed_name = census_constants.transformed_name

def _gzip_reader_fn(filenames):  
    # Load the dataset. Specify the compression type since it is saved as `.gz`
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def _input_fn(file_pattern,
              tf_transform_output,
              num_epochs=None,
              batch_size=32) -> tf.data.Dataset:
    
    transformed_feature_spec = (
      tf_transform_output.transformed_feature_spec().copy())
    
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=_transformed_name(_LABEL_KEY)
    )
    
    return dataset

def _get_serve_tf_examples_fn(model, tf_transform_output):
    
    # Get transformation graph
    model.tft_layer = tf_transform_output.transform_features_layer()
       
    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        # Get pre-transform feature spec
        feature_spec = tf_transform_output.raw_feature_spec()

        # Pop label since serving inputs do not include the label
        feature_spec.pop(_LABEL_KEY)

        # Parse raw examples into a dictionary of tensors matching the feature spec
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

        # Transform the raw examples using the transform graph
        transformed_features = model.tft_layer(parsed_features)

        # Get predictions using the transformed features
        return model(transformed_features)

    return serve_tf_examples_fn


def _make_keras_model(hp) -> tf.keras.Model:
    # Define input layers for numeric keys
    input_numeric = [
        tf.keras.layers.Input(name=_transformed_name(colname), shape=(1,), dtype=tf.float32)
        for colname in _NUMERIC_FEATURE_KEYS
    ]
    
    # Define input layers for vocab keys
    input_categorical = [
        tf.keras.layers.Input(name=_transformed_name(colname), shape=(vocab_size,), dtype=tf.float32)
        for colname, vocab_size in _VOCAB_FEATURE_DICT.items()
    ]
    
    # Define input layers for bucket key
    input_categorical += [
        tf.keras.layers.Input(name=_transformed_name(colname), shape=(num_buckets,), dtype=tf.float32)
        for colname, num_buckets in _BUCKET_FEATURE_DICT.items()
    ]
    
    # Concatenate numeric inputs
    deep = tf.keras.layers.concatenate(input_numeric)
    
    # Create deep dense network for numeric inputs
    hp_units = hp.get('units')
    for _ in range(4):
        deep = keras.layers.Dense(hp_units, activation='relu')(deep)
    
    # Concatenate categorical inputs
    wide = tf.keras.layers.concatenate(input_categorical)
    
    # Combine wide and deep networks
    combined = tf.keras.layers.concatenate([deep, wide])
    
    # Define output of binary classifier
    output = tf.keras.layers.Dense(1, activation='sigmoid')(combined)
    
    # Setup combined input
    input_layers = input_numeric + input_categorical
    
    # Create the Keras model
    model = tf.keras.Model(input_layers, output)
    
    # Define training parameters
    hp_learning_rate = hp.get('learning_rate')
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        metrics='accuracy')
    
    return model
        
def run_fn(fn_args: FnArgs) -> None:
    
    # Callback for TensorBoard
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir, 
        update_freq='batch')
    
    # Load transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
  
    # Create batches of data good for 10 epochs
    train_set = _input_fn(fn_args.train_files[0], tf_transform_output, 10)
    val_set = _input_fn(fn_args.eval_files[0], tf_transform_output, 10)
    
    # Load best hyperparameters
    hp = fn_args.hyperparameters.get('values')
    
    # Build the model
    model = _make_keras_model(hp) #model_builder()
    
    # Train the model
    
    model.fit(
        x=train_set,
        steps_per_epoch=fn_args.train_steps,
        validation_data=val_set,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback]
      )
    
    # Define default serving signature
    signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model,
                                    tf_transform_output).get_concrete_function(
                                        tf.TensorSpec(
                                            shape=[None],
                                            dtype=tf.string,
                                            name='examples')),
    }

    # Save the model
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
