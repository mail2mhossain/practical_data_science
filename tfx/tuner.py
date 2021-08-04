
# Define imports
from kerastuner.engine import base_tuner
import kerastuner as kt
from tensorflow import keras
from typing import NamedTuple, Dict, Text, Any, List
from tfx.components.trainer.fn_args_utils import FnArgs, DataAccessor
import tensorflow as tf
import tensorflow_transform as tft

# import same constants from transform module
import census_constants

# Unpack the contents of the constants module
_NUMERIC_FEATURE_KEYS = census_constants.NUMERIC_FEATURE_KEYS
_VOCAB_FEATURE_DICT = census_constants.VOCAB_FEATURE_DICT
_BUCKET_FEATURE_DICT = census_constants.BUCKET_FEATURE_DICT
_NUM_OOV_BUCKETS = census_constants.NUM_OOV_BUCKETS
_LABEL_KEY = census_constants.LABEL_KEY
_transformed_name = census_constants.transformed_name

TunerFnResult = NamedTuple('TunerFnResult', [('tuner', base_tuner.BaseTuner),
                                             ('fit_kwargs', Dict[Text, Any])])

# Callback for the search strategy
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)


def _gzip_reader_fn(filenames):
    # Load the dataset. Specify the compression type since it is saved as `.gz`
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def _input_fn(file_pattern,
              tf_transform_output,
              num_epochs=None,
              batch_size=32) -> tf.data.Dataset:
    
    transformed_feature_spec = (
      tf_transform_output.transformed_feature_spec().copy())
    
    # Create batches of features and labels
    dataset = tf.data.experimental.make_batched_features_dataset(
      file_pattern=file_pattern,
      batch_size=batch_size,
      features=transformed_feature_spec,
      reader=_gzip_reader_fn,
      num_epochs=num_epochs,
      label_key=_transformed_name(_LABEL_KEY))
    
    return dataset


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
    hp_units = hp.Int('units', min_value=32, max_value=512, step=16)
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
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(
        loss='binary_crossentropy',
        #optimizer=tf.keras.optimizers.Adam(lr=0.001),
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        metrics='accuracy')
    
    return model
    

def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    
    # Define tuner search strategy
    tuner = kt.Hyperband(_make_keras_model,
                     objective='accuracy', 
                     max_epochs=50,
                     factor=3,
                     directory=fn_args.working_dir,
                     project_name='kt_hyperband')
    
    # Load transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
        
    # Use _input_fn() to extract input features and labels from the train and val set
    train_set = _input_fn(fn_args.train_files[0], tf_transform_output)
    val_set = _input_fn(fn_args.eval_files[0], tf_transform_output)
    
    
    return TunerFnResult(
      tuner=tuner,
      fit_kwargs={ 
          "callbacks":[stop_early],
          'x': train_set,
          'validation_data': val_set,
          'steps_per_epoch': fn_args.train_steps,
          'validation_steps': fn_args.eval_steps
      }
    )
