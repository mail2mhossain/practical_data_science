
# Features with string data types that will be converted to indices
VOCAB_FEATURE_DICT = {
    'education': 16, 'marital-status': 7, 'occupation': 15, 'race': 5, 
    'relationship': 6, 'workclass': 9, 'sex': 2, 'native-country': 42
}

# Numerical features that are marked as continuous
NUMERIC_FEATURE_KEYS = ['fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

# Feature that can be grouped into buckets
BUCKET_FEATURE_DICT = {'age': 4}

# Number of out-of-vocabulary buckets
NUM_OOV_BUCKETS = 5

# Feature that the model will predict
LABEL_KEY = 'label'

# Utility function for renaming the feature
def transformed_name(key):
    return key + '_xf'
