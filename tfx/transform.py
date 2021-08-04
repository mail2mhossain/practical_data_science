
import tensorflow as tf
import tensorflow_transform as tft

# import constants from cells above
import census_constants

# Unpack the contents of the constants module
_NUMERIC_FEATURE_KEYS = census_constants.NUMERIC_FEATURE_KEYS
_VOCAB_FEATURE_DICT = census_constants.VOCAB_FEATURE_DICT
_BUCKET_FEATURE_DICT = census_constants.BUCKET_FEATURE_DICT
_NUM_OOV_BUCKETS = census_constants.NUM_OOV_BUCKETS
_LABEL_KEY = census_constants.LABEL_KEY
_transformed_name = census_constants.transformed_name


def preprocessing_fn(inputs):
    outputs = {}

    # Scale these features to the range [0,1]
    for key in _NUMERIC_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.scale_to_0_1(inputs[key])
    
    # Bucketize these features
    for key, num_buckets in _BUCKET_FEATURE_DICT.items():
        #outputs[transformed_name(key)] = tft.bucketize(inputs[key], FEATURE_BUCKET_COUNT[key])
        indices = tft.bucketize(inputs[key], num_buckets)
        one_hot = tf.one_hot(indices, num_buckets)
        outputs[_transformed_name(key)] = tf.reshape(one_hot, [-1, num_buckets])
            #always_return_num_quantiles=False)

    # Convert strings to indices in a vocabulary
    for key, vocab_size in _VOCAB_FEATURE_DICT.items():
        #outputs[transformed_name(key)] = tft.compute_and_apply_vocabulary(inputs[key])
        indices = tft.compute_and_apply_vocabulary(inputs[key])
        one_hot = tf.one_hot(indices, vocab_size)
        outputs[_transformed_name(key)] = tf.reshape(one_hot, [-1, vocab_size])

    # Convert the label strings to an index
    outputs[_transformed_name(_LABEL_KEY)] = tft.compute_and_apply_vocabulary(inputs[_LABEL_KEY])

    return outputs
